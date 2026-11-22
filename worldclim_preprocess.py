import numpy as np
import pandas as pd
from global_land_mask import globe
#pip install rasterio
import rasterio
import rasterio.features
import rasterio.warp
import h3

def near_land(lat, lon, step=.25): # .25 d- about 28km N-S
    near = np.zeros(len(lat), dtype=bool)
    for dlat in (-step, 0, step):
        for dlon in (-step, 0, step):
            clip_lat = np.clip(lat+dlat,-90,90)
            clip_long = np.clip(lon+dlon,-180,180)
            near |= globe.is_land(clip_lat, clip_long)
    return near

# Features- normalise lat/lon to [-1,1], then sin/cos (SINR paper approach)
def encode(latlon):
    lat = latlon[:,0]/90
    lon = latlon[:,1]/180
    return np.column_stack([np.sin(np.pi*lon), np.cos(np.pi*lon), np.sin(np.pi*lat), np.cos(np.pi*lat)])

def convert_tif_to_lonlat(file_name, bio_clim):
    
    data = rasterio.open(f"wc2.1_10m_bio/{file_name}")

    band1 = data.read(1)
    
    # create columns and rows (0 to 1079 and 0 to 2159)
    col = np.arange(data.width)
    row = np.arange(data.height)
    
    # create the column and row grid
    col_grid, row_grid = np.meshgrid(col, row)
    
    # take the latitude and longitude (add by 0.5 to get the center and not the top left corner)
    lon, lat = data.transform * (col_grid + 0.5, row_grid + 0.5)
    
    data_bio = []
    for i in row:
        for j in col:
            data_bio.append([lon[i, j], lat[i, j], band1[i, j]])
    
    data_bio = pd.DataFrame(data_bio, columns=['lon', 'lat', bio_clim])

    data.close()

    return data_bio

# Load all selected bioclim data
data_bio_1 = convert_tif_to_lonlat('wc2.1_10m_bio_1.tif', 'Annual Mean Temperature')
data_bio_4 = convert_tif_to_lonlat('wc2.1_10m_bio_4.tif', 'Temperature Seasonality')
data_bio_5 = convert_tif_to_lonlat('wc2.1_10m_bio_5.tif', 'Max Temperature of Warmest Month')
data_bio_6 = convert_tif_to_lonlat('wc2.1_10m_bio_6.tif', 'Min Temperature of Coldest Month')
data_bio_12 = convert_tif_to_lonlat('wc2.1_10m_bio_12.tif', 'Annual Precipitation')
data_bio_15 = convert_tif_to_lonlat('wc2.1_10m_bio_15.tif', 'Precipitation Seasonality')
data_bio_17 = convert_tif_to_lonlat('wc2.1_10m_bio_17.tif', 'Precipitation of Driest Quarter')
elev_data = convert_tif_to_lonlat('wc2.1_10m_elev.tif', 'Elevation')

# Column names
col_names = ['Annual Mean Temperature',
            'Temperature Seasonality',
             'Max Temperature of Warmest Month',
             'Min Temperature of Coldest Month',
             'Annual Precipitation',
             'Precipitation Seasonality',
             'Precipitation of Driest Quarter',
             'Elevation']

data_var = [data_bio_1, data_bio_4, data_bio_5, data_bio_6, data_bio_12, data_bio_15, data_bio_17, elev_data]

# make one table for all the bioclim data
data_bioclim = data_bio_1[['lon', 'lat']].copy()
for i in range(len(col_names)):
    col = col_names[i]
    var = data_var[i]
    data_bioclim[col] = var[col].values

# Remove worldclim cells- about 25km from land
data_bioclim = data_bioclim[near_land(data_bioclim['lat'].values, data_bioclim['lon'].values)].copy()

# drop the data that have no values (WorldClim uses -3.4e38 (bioclim) and -32768 (elevation))
data_bioclim[col_names] = data_bioclim[col_names].mask(data_bioclim[col_names] < -9999)
data_bioclim = data_bioclim.dropna(subset=col_names)

# remove polar edge cells with zero temperature seasonality
data_bioclim = data_bioclim[data_bioclim['Temperature Seasonality'] > 0]


data_bioclim[['sin_lon', 'cos_lon', 'sin_lat', 'cos_lat']] = encode(data_bioclim[['lat', 'lon']].values)

# Normalize the bioclim data
for i in range(len(col_names)):
    col = col_names[i]
    mu = data_bioclim[col].mean()
    sigma = data_bioclim[col].std()
    data_bioclim[col] = (data_bioclim[col] - mu) / sigma

res = 4

# Grid- every H3 cell on the globe
grid_cells = [c for r0 in h3.get_res0_cells() for c in h3.cell_to_children(r0, res)]
grid = np.array([h3.cell_to_latlng(c) for c in grid_cells])
cell_index = {c: i for i, c in enumerate(grid_cells)}

# H3 cell of every WorldClim cell
data_bioclim['cell'] = [cell_index[h3.latlng_to_cell(lat, lon, res)] for lat, lon in zip(data_bioclim['lat'], data_bioclim['lon'])]

# Average per hexagon
bioclim_h3 = data_bioclim.groupby('cell')[col_names].mean()

# One row per grid cell, same order as grid (NaN = no climate data)
env_grid = bioclim_h3.reindex(np.arange(len(grid))).values

np.savez('bioclim_h3_res4.npz',
         env_grid=env_grid,
         grid_cells=np.array(grid_cells),
         env_names=np.array(col_names),
         res=res)

print(env_grid)

# Load the preprocessed cells
preprocessed = np.load('species/preprocessed_cells.npz')
y_train = preprocessed['y_train']
absent_train = preprocessed['absent_train']

cols = ['sin_lon',
        'cos_lon',
        'sin_lat',
        'cos_lat'
       ]

# Column names
env_cols = ['Annual Mean Temperature',
            'Temperature Seasonality',
             'Max Temperature of Warmest Month',
             'Min Temperature of Coldest Month',
             'Annual Precipitation',
             'Precipitation Seasonality',
             'Precipitation of Driest Quarter',
             'Elevation']

# Make the train df from the preprocessed cells
train_df_bio = pd.DataFrame(preprocessed['X_train'], columns=cols)
train_df_bio['cell'] = preprocessed['cell_train']

# climate for each cell
train_df_bio[env_cols] = env_grid[train_df_bio['cell'].values]

# Drop cells with no climate data (ocean)
clean_bio = train_df_bio[env_cols].notna().all(axis=1).values
train_df_bio = train_df_bio[clean_bio]

X_train_bio = train_df_bio[cols + env_cols].values
y_train_bio = y_train[clean_bio]
absent_train_bio = absent_train[clean_bio]

# Save
np.savez('preprocessed_cells_bio.npz', X_train=X_train_bio, y_train=y_train_bio, absent_train=absent_train_bio)
print('X_train_bio-', X_train_bio.shape, ', y_train_bio-', y_train_bio.shape)