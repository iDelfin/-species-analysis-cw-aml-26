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


def convert_future_tif_to_lonlat(file_name, bio_clim, index):
    
    data = rasterio.open(file_name)

    band = data.read(index)
    
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
            data_bio.append([lon[i, j], lat[i, j], band[i, j]])
    
    data_bio = pd.DataFrame(data_bio, columns=['lon', 'lat', bio_clim])

    data.close()

    return data_bio


# Historical mean and std (saved by the historical WorldClim script)
stats = np.load('bioclim_norm_stats.npz')
hist_mu = stats['mu']
hist_sigma = stats['sigma']

# Load all selected bioclim data
future_file = 'wc2.1_10m_bioc_MPI-ESM1-2-HR_ssp245_2041-2060.tif'

future_data_bio_1  = convert_future_tif_to_lonlat(future_file, 'Future Annual Mean Temperature', 1)
future_data_bio_4  = convert_future_tif_to_lonlat(future_file, 'Future Temperature Seasonality', 4)
future_data_bio_5  = convert_future_tif_to_lonlat(future_file, 'Future Max Temperature of Warmest Month', 5)
future_data_bio_6  = convert_future_tif_to_lonlat(future_file, 'Future Min Temperature of Coldest Month', 6)
future_data_bio_12 = convert_future_tif_to_lonlat(future_file, 'Future Annual Precipitation', 12)
future_data_bio_15 = convert_future_tif_to_lonlat(future_file, 'Future Precipitation Seasonality', 15)
future_data_bio_17 = convert_future_tif_to_lonlat(future_file, 'Future Precipitation of Driest Quarter', 17)

# use same elevation as before because it does not change
elev_data = convert_future_tif_to_lonlat('wc2.1_10m_bio/wc2.1_10m_elev.tif', 'Elevation', 1)


# Column names
future_col_names = ['Future Annual Mean Temperature',
                    'Future Temperature Seasonality',
                    'Future Max Temperature of Warmest Month',
                    'Future Min Temperature of Coldest Month',
                    'Future Annual Precipitation',
                    'Future Precipitation Seasonality',
                    'Future Precipitation of Driest Quarter',
                    'Elevation']

future_data_var = [future_data_bio_1, future_data_bio_4, future_data_bio_5, future_data_bio_6, future_data_bio_12, future_data_bio_15, future_data_bio_17, elev_data]

# make one table for all the bioclim data
future_data_bioclim = future_data_bio_1[['lon', 'lat']].copy()
for i in range(len(future_col_names)):
    col = future_col_names[i]
    var = future_data_var[i]
    future_data_bioclim[col] = var[col].values

# Remove worldclim cells- about 25km from land
future_data_bioclim = future_data_bioclim[near_land(future_data_bioclim['lat'].values, future_data_bioclim['lon'].values)].copy()

# drop the data that have no values (WorldClim uses -3.4e38 (bioclim) and -32768 (elevation))
future_data_bioclim[future_col_names] = future_data_bioclim[future_col_names].mask(future_data_bioclim[future_col_names] < -9999)
future_data_bioclim = future_data_bioclim.dropna(subset=future_col_names)

# remove polar edge cells with zero temperature seasonality
future_data_bioclim = future_data_bioclim[future_data_bioclim['Future Temperature Seasonality'] > 0]


future_data_bioclim[['sin_lon', 'cos_lon', 'sin_lat', 'cos_lat']] = encode(future_data_bioclim[['lat', 'lon']].values)

# Normalise with the historical mean and std to preserve climate change
for i in range(len(future_col_names)):
    col = future_col_names[i]
    future_data_bioclim[col] = (future_data_bioclim[col] - hist_mu[i]) / hist_sigma[i]

res = 4

# Grid- every H3 cell on the globe
grid_cells = [c for r0 in h3.get_res0_cells() for c in h3.cell_to_children(r0, res)]
grid = np.array([h3.cell_to_latlng(c) for c in grid_cells])
cell_index = {c: i for i, c in enumerate(grid_cells)}

# H3 cell of every WorldClim cell
future_data_bioclim['cell'] = [cell_index[h3.latlng_to_cell(lat, lon, res)] for lat, lon in zip(future_data_bioclim['lat'], future_data_bioclim['lon'])]

# Average per hexagon
future_bioclim_h3 = future_data_bioclim.groupby('cell')[future_col_names].mean()

# One row per grid cell, same order as grid (NaN = no climate data)
future_env_grid = future_bioclim_h3.reindex(np.arange(len(grid))).values

np.savez('future_bioclim_h3_res4.npz',
         future_env_grid=future_env_grid,
         grid_cells=np.array(grid_cells),
         future_env_names=np.array(future_col_names),
         res=res)

print(future_env_grid)

### LINK WITH THE TEST DATA #####
# Load the preprocessed test cells
preprocessed_test = np.load('species/preprocessed_test.npz')
X_test = preprocessed_test['X_test']
y_test = preprocessed_test['y_test']
cell_test = preprocessed_test['cell_test']

cols = ['sin_lon',
        'cos_lon',
        'sin_lat',
        'cos_lat'
       ]

# Column names
future_col_names = ['Future Annual Mean Temperature',
                    'Future Temperature Seasonality',
                    'Future Max Temperature of Warmest Month',
                    'Future Min Temperature of Coldest Month',
                    'Future Annual Precipitation',
                    'Future Precipitation Seasonality',
                    'Future Precipitation of Driest Quarter',
                    'Elevation']

# Load the future climate grid
future_env_grid = np.load('future_bioclim_h3_res4.npz')['future_env_grid']

# Make the test df from the preprocessed test cells
test_df_future = pd.DataFrame(X_test, columns=cols)
test_df_future['cell'] = cell_test

# climate for each cell
test_df_future[future_col_names] = future_env_grid[test_df_future['cell'].values]

# Drop cells with no climate data (ocean)
clean_future = test_df_future[future_col_names].notna().all(axis=1).values
test_df_future = test_df_future[clean_future]

X_test_future = test_df_future[cols + future_col_names].values
y_test_future = y_test[clean_future]
cell_test_future = cell_test[clean_future]

# Save
np.savez('preprocessed_test_future.npz', X_test=X_test_future, y_test=y_test_future, cell_test=cell_test_future)
print('X_test_future-', X_test_future.shape, ', y_test_future-', y_test_future.shape)