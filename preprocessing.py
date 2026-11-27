"""
Run pip install -r requirements.txt to install the required packages before this script.
1.   All Imports
"""

import numpy as np
import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor
from global_land_mask import globe
import h3
import rasterio

"""## Preprocessing

1.   Taxonomy CSV
2.   Preprocess Training Data
3.   Preprocess Test Data
4.   Preprocess Climate Data
5.   Preprocess Future Climate Data
"""



"""taxonomy_species.py"""



GBIF = 'https://api.gbif.org/v1/species/match'
GROUPS = {'Aves': 'bird', 'Mammalia': 'mammal', 'Amphibia': 'amphibian', 'Reptilia': 'reptile', 'Squamata': 'reptile', 'Testudines': 'reptile', 'Crocodylia': 'reptile'}

def gbif_class(name):
    for query in (name, name.split()[0]): # species name first, then genus, incase it doesn't recognise
        result = requests.get(GBIF, params={'name': query, 'kingdom': 'Animalia'}, timeout=30).json()
        if result.get('class'):
            return result['class']
    return None

# Species from both files (in case we use the extra data too)
rows = []
for source in ['species_train', 'species_train_extra']:
    data = np.load(f'species/{source}.npz')
    for taxon_id, name in zip(data['taxon_ids'], data['taxon_names']):
        rows.append({'taxon_id': taxon_id, 'name': name, 'source': source})
df = pd.DataFrame(rows)
print('Species-', len(df))

# Look up on GBIF, using ThreadPoolExecutor to send 10 req at a time
with ThreadPoolExecutor(10) as pool:
    df['class'] = list(pool.map(gbif_class, df['name']))
df['group'] = df['class'].map(GROUPS)

df = df[['taxon_id', 'name', 'source', 'group', 'class']]
df.to_csv('species/species_taxonomy.csv', index=False)
print(df['class'].value_counts(dropna=False))



"""preprocess_cells.py"""



import numpy as np
import pandas as pd
from global_land_mask import globe
import h3

SEED=0
res=4
Abs_per_presence=1

# Load training data
train = np.load('species/species_train.npz')

# Grid- every H3 cell on the globe
grid_cells = [c for r0 in h3.get_res0_cells() for c in h3.cell_to_children(r0, res)]
grid = np.array([h3.cell_to_latlng(c) for c in grid_cells]) # centers
cell_index = {c: i for i, c in enumerate(grid_cells)}
print('Grid cells-', len(grid))

df = pd.DataFrame({'lat': train['train_locs'][:, 0], 'lon': train['train_locs'][:, 1], 'species': train['train_ids'],})
print('Sightings-', len(df))

def near_land(lat, lon, step=.25): # .25 d- about 28km N-S
    near = np.zeros(len(lat), dtype=bool)
    for dlat in (-step, 0, step):
        for dlon in (-step, 0, step):
            clip_lat = np.clip(lat+dlat,-90,90)
            clip_long = np.clip(lon+dlon,-180,180)
            near |= globe.is_land(clip_lat, clip_long)
    return near

# Remove ocean sightings- about 25km from land
df = df[near_land(df['lat'].values, df['lon'].values)].copy()
print('Total after removing ocean sightings-', len(df))
land_cells = np.flatnonzero(near_land(grid[:, 0], grid[:, 1]))
print('Land cells-', len(land_cells))

# Bin- find the H3 cell each sighting falls in
df['cell'] = [cell_index[h3.latlng_to_cell(lat, lon, res)] for lat, lon in zip(df['lat'], df['lon'])]

# Index for species 0-499
species_index = {taxon: i for i, taxon in enumerate(train['taxon_ids'])}
df['index'] = df['species'].map(species_index)

# Presence only- one row per cell, list of species seen there
presences = df.groupby('cell')['index'].unique().apply(sorted).reset_index(name='species')
presences['n_sightings'] = df.groupby('cell').size().values
print('Cells with sightings-', len(presences))

# Pseudo-absences- per species
rng = np.random.default_rng(SEED)
pairs = df[['cell', 'species']].drop_duplicates()  # one row per (cell, species) presence
absent_cells = []
absent_species = []
for species, group in pairs.groupby('species'):
    candidates = np.setdiff1d(land_cells, group['cell'])  # Pseudo-absences in cells where presences are not there
    picked = rng.choice(candidates, len(group)*Abs_per_presence, replace=False)
    absent_cells.extend(picked)
    absent_species.extend([species_index[species]] * len(picked))

# One row per cell, with the list of species pseudo-absence
absences = pd.DataFrame({'cell': absent_cells, 'absent': absent_species})
absences = absences.groupby('cell')['absent'].apply(list).reset_index()
absences['species'] = [[] for _ in range(len(absences))]  # no species seen
absences['n_sightings'] = 0

data = pd.concat([presences, absences], ignore_index=True)
data['absent'] = [a if type(a) == list else [] for a in data['absent']]
data['n_species'] = data['species'].apply(len)
print('Pseudo-absences-', len(absent_cells)) # Number of pseudo absences

# Taxonomy class for species
taxonomy = pd.read_csv('species/species_taxonomy.csv')
species_class = dict(zip(taxonomy['taxon_id'], taxonomy['class']))

# Number for each class
# {0: 'Amphibia', 1: 'Aves', 2: 'Crocodylia', 3: 'Mammalia', 4: 'Squamata', 5: 'Testudines'}
class_names = sorted(taxonomy['class'].unique())
class_index = {}
for i, c in enumerate(class_names):
    class_index[c] = i
class_id = []
for t in train['taxon_ids']:
    class_id.append(class_index[species_class[t]])
class_id = np.array(class_id)

# Features- normalise lat/lon to [-1,1], then sin/cos (SINR paper approach)
def encode(latlon):
    lat = latlon[:,0]/90
    lon = latlon[:,1]/180
    return np.column_stack([np.sin(np.pi*lon), np.cos(np.pi*lon), np.sin(np.pi*lat), np.cos(np.pi*lat)])

X_train = encode(grid[data['cell']])
y_train = np.zeros((len(data), len(species_index)), dtype=bool) # 1=species present
for row, species in enumerate(data['species']):
    y_train[row, species] = 1

absent_train = np.zeros((len(data), len(species_index)), dtype=bool)
for row, absent in enumerate(data['absent']):
    absent_train[row, absent] = 1

# Save
np.savez('species/preprocessed_cells.npz', X_train=X_train, y_train=y_train, absent_train=absent_train, class_id=class_id, n_sightings=data['n_sightings'].values, n_species=data['n_species'].values, cell_train=data['cell'].values, grid_latlon=grid, grid_cells=np.array(grid_cells), res=res, taxon_ids=train['taxon_ids'])
print('X_train-', X_train.shape, ', y_train-', y_train.shape)

# Usage for models
preprocessed = np.load('species/preprocessed_cells.npz')
cols = ['sin_lon', 'cos_lon', 'sin_lat', 'cos_lat']
train_df = pd.DataFrame(preprocessed['X_train'], columns=cols)
train_df['cell'] = preprocessed['cell_train']
train_df['n_species'] = preprocessed['n_species']
train_df['sightings'] = preprocessed['n_sightings']
train_df['species'] = [np.flatnonzero(row) for row in preprocessed['y_train']] # list view

X = preprocessed['X_train']
y = preprocessed['y_train'].astype(np.float32) # for NN, 0/1 col for each grid cell for all 500 species

# NaN 1=present, 0=absent, NaN=unknown
y_nan = y.copy()
y_nan[(y_nan == 0) & ~preprocessed['absent_train']] = np.nan
print('present-', (y_nan == 1).sum(), 'absent-', (y_nan == 0).sum(), 'unknown-', np.isnan(y_nan).sum())

print(train_df.head())



"""preprocess_species.py"""



SEED=0
res=4
Abs_per_presence=1

# Load training data
train = np.load('species/species_train.npz')

# Grid- every H3 cell on the globe
grid_cells = [c for r0 in h3.get_res0_cells() for c in h3.cell_to_children(r0, res)]
grid = np.array([h3.cell_to_latlng(c) for c in grid_cells]) # centers
cell_index = {c: i for i, c in enumerate(grid_cells)}
print('Grid cells-', len(grid))

df = pd.DataFrame({'lat': train['train_locs'][:, 0], 'lon': train['train_locs'][:, 1], 'species': train['train_ids'],})
print('Sightings-', len(df))

def near_land(lat, lon, step=.25): # .25 d- about 28km N-S
    near = np.zeros(len(lat), dtype=bool)
    for dlat in (-step, 0, step):
        for dlon in (-step, 0, step):
            clip_lat = np.clip(lat+dlat,-90,90)
            clip_long = np.clip(lon+dlon,-180,180)
            near |= globe.is_land(clip_lat, clip_long)
    return near

# Remove ocean sightings- about 25km from land
df = df[near_land(df['lat'].values, df['lon'].values)].copy()
print('Total after removing ocean sightings-', len(df))
land_cells = np.flatnonzero(near_land(grid[:, 0], grid[:, 1]))
print('Land cells-', len(land_cells))

# Bin- find the H3 cell each sighting falls in
df['cell'] = [cell_index[h3.latlng_to_cell(lat, lon, res)] for lat, lon in zip(df['lat'], df['lon'])]

# Presence only- one row per (cell, species)
presences = df.groupby(['cell', 'species']).size().reset_index(name='n_sightings') # Keep number of sightings
presences['present'] = 1
print('Presences after binning-', len(presences))

# Pseudo-absences- random grid cells (not same as presences but could be in the neighboring cell)
rng = np.random.default_rng(SEED)
absences = []
for species, group in presences.groupby('species'):
    candidates = np.setdiff1d(land_cells, group['cell']) # Pseudo-absences in cells where presences are not there
    picked = rng.choice(candidates, len(group)*Abs_per_presence, replace=False)
    absences.append(pd.DataFrame({'cell': picked, 'species': species, 'present': 0, 'n_sightings': 0}))

data = pd.concat([presences]+absences, ignore_index=True)
print('Pseudo-absences-', (data['present']==0).sum())

# Index for species 0-499
species_index = {taxon: i for i, taxon in enumerate(train['taxon_ids'])}
data['index'] = data['species'].map(species_index)

# Taxonomy class for species
taxonomy = pd.read_csv('species/species_taxonomy.csv')
species_class = dict(zip(taxonomy['taxon_id'], taxonomy['class']))
data['class'] = data['species'].map(species_class)

# Number for each class
# {0: 'Amphibia', 1: 'Aves', 2: 'Crocodylia', 3: 'Mammalia', 4: 'Squamata', 5: 'Testudines'}
class_names = sorted(taxonomy['class'].unique())
class_index = {}
for i, c in enumerate(class_names):
    class_index[c] = i
data['class_id'] = data['class'].map(class_index)

# Features- normalise lat/lon to [-1,1], then sin/cos (SINR paper approach)
def encode(latlon):
    lat = latlon[:,0]/90
    lon = latlon[:,1]/180
    return np.column_stack([np.sin(np.pi*lon), np.cos(np.pi*lon), np.sin(np.pi*lat), np.cos(np.pi*lat)])

X_train = encode(grid[data['cell']])
y_train = data['present'].values
species_train = data['species'].values

# Save
np.savez('species/preprocessed_species.npz', id=data['index'].values, X_train=X_train, y_train=y_train, species_train=species_train, class_id=data['class_id'].values, n_sightings=data['n_sightings'].values, cell_train=data['cell'].values, grid_latlon=grid, grid_cells=np.array(grid_cells), res=res, taxon_ids=train['taxon_ids'])
print('X_train-', X_train.shape, ', y_train-', y_train.shape)

# Usage for models
preprocessed = np.load('species/preprocessed_species.npz')
cols = ['sin_lon', 'cos_lon', 'sin_lat', 'cos_lat']
train_df = pd.DataFrame(preprocessed['X_train'], columns=cols)
train_df['id'] = preprocessed['id']
train_df['class_id'] = preprocessed['class_id']
train_df['species'] = preprocessed['species_train']
train_df['present'] = preprocessed['y_train']
train_df['sightings'] = preprocessed['n_sightings']

X = preprocessed['X_train']
y = preprocessed['y_train']

print(train_df.head())



"""preprocess_cells_test.py"""



import numpy as np
import h3
import pandas as pd

test = np.load('species/species_test.npz', allow_pickle=True)

# Features- normalise lat/lon to [-1,1], then sin/cos (SINR paper approach)
def encode(latlon):
    lat = latlon[:,0]/90
    lon = latlon[:,1]/180
    return np.column_stack([np.sin(np.pi*lon), np.cos(np.pi*lon), np.sin(np.pi*lat), np.cos(np.pi*lat)])

res = 4
grid_cells = [c for r0 in h3.get_res0_cells() for c in h3.cell_to_children(r0, res)]
cell_index = {c: i for i, c in enumerate(grid_cells)}
cell_test= []
for lat, lon in test['test_locs'].astype(float):
    cell_test.append(cell_index[h3.latlng_to_cell(lat, lon, res)])
cell_test = np.array(cell_test)

X_test = encode(test['test_locs'])
y_test = np.zeros((len(X_test), len(test['taxon_ids'])), dtype=bool)
for i, pos in enumerate(test['test_pos_inds']):
    y_test[pos, i] = 1  # species i present at these locations, 0 elsewhere

print('X_test-', X_test.shape, ', y_test-', y_test.shape, ', cell_test-', cell_test.shape)

# Save
np.savez('species/preprocessed_cells_test.npz', X_test=X_test, y_test=y_test, cell_test=cell_test, test_locs=test['test_locs'], taxon_ids=test['taxon_ids'])

# Usage for models
preprocessed = np.load('species/preprocessed_cells_test.npz')
cols = ['sin_lon', 'cos_lon', 'sin_lat', 'cos_lat']
test_df = pd.DataFrame(preprocessed['X_test'], columns=cols)
test_df['cell'] = preprocessed['cell_test']
test_df['n_species'] = preprocessed['y_test'].sum(axis=1)  # number of species present at each point
test_df['species'] = [np.flatnonzero(row) for row in preprocessed['y_test']]

X = preprocessed['X_test']
y = preprocessed['y_test'].astype(np.float32)  # for NN, 0/1 for each test point for all 500 species

print(test_df.head())



"""preprocess_species_test.py"""



import numpy as np
import pandas as pd
import h3

res=4

# Load test data
test = np.load('species/species_test.npz', allow_pickle=True)
print('Test points-', len(test['test_locs']))

# Grid- every H3 cell on the globe
grid_cells = [c for r0 in h3.get_res0_cells() for c in h3.cell_to_children(r0, res)]
cell_index = {c: i for i, c in enumerate(grid_cells)}

# H3 cell of each test point
cell_all = []
for lat, lon in test['test_locs'].astype(float):
    cell_all.append(cell_index[h3.latlng_to_cell(lat, lon, res)])
cell_all = np.array(cell_all)

# Presence only- one row per (test point, species) where the species is present
presences = []
for i, pos in enumerate(test['test_pos_inds']):
    presences.append(pd.DataFrame({'point': pos, 'index': i}))
data = pd.concat(presences, ignore_index=True)
data['species'] = test['taxon_ids'][data['index']]
data['present'] = 1
data['cell'] = cell_all[data['point']]
print('Presences-', len(data))

# Taxonomy class for species
taxonomy = pd.read_csv('species/species_taxonomy.csv')
species_class = dict(zip(taxonomy['taxon_id'], taxonomy['class']))
data['class'] = data['species'].map(species_class)

# Number for each class
# {0: 'Amphibia', 1: 'Aves', 2: 'Crocodylia', 3: 'Mammalia', 4: 'Squamata', 5: 'Testudines'}
class_names = sorted(taxonomy['class'].unique())
class_index = {}
for i, c in enumerate(class_names):
    class_index[c] = i
data['class_id'] = data['class'].map(class_index)

# Features- normalise lat/lon to [-1,1], then sin/cos (SINR paper approach)
def encode(latlon):
    lat = latlon[:,0]/90
    lon = latlon[:,1]/180
    return np.column_stack([np.sin(np.pi*lon), np.cos(np.pi*lon), np.sin(np.pi*lat), np.cos(np.pi*lat)])

X_test = encode(test['test_locs'][data['point']])
y_test = data['present'].values
species_test = data['species'].values

# All test points
X_all = encode(test['test_locs'])

# Save
np.savez('species/preprocessed_species_test.npz', id=data['index'].values, X_test=X_test, y_test=y_test, species_test=species_test, class_id=data['class_id'].values, cell_test=data['cell'].values, point_test=data['point'].values, X_all=X_all, cell_all=cell_all, test_locs=test['test_locs'], taxon_ids=test['taxon_ids'])
print('X_test-', X_test.shape, ', y_test-', y_test.shape, ', X_all-', X_all.shape)

# Usage for models
preprocessed = np.load('species/preprocessed_species_test.npz')
cols = ['sin_lon', 'cos_lon', 'sin_lat', 'cos_lat']
test_df = pd.DataFrame(preprocessed['X_test'], columns=cols)
test_df['id'] = preprocessed['id']
test_df['class_id'] = preprocessed['class_id']
test_df['species'] = preprocessed['species_test']
test_df['present'] = preprocessed['y_test']
test_df['cell'] = preprocessed['cell_test']

print(test_df.head())



"""worldclim_cells.py"""



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

    data = rasterio.open(f"climate/wc2.1_10m_bio/{file_name}")

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
# save the historical mu and sigma so the future data can be normalised the same way
hist_mu = []
hist_sigma = []
for i in range(len(col_names)):
    col = col_names[i]
    mu = data_bioclim[col].mean()
    sigma = data_bioclim[col].std()
    hist_mu.append(mu)
    hist_sigma.append(sigma)
    data_bioclim[col] = (data_bioclim[col] - mu) / sigma

np.savez('climate/bioclim_norm_stats.npz', mu=np.array(hist_mu), sigma=np.array(hist_sigma), env_names=np.array(col_names))

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

np.savez('climate/bioclim_h3_res4.npz',
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
np.savez('climate/preprocessed_cells_bio.npz', X_train=X_train_bio, y_train=y_train_bio, absent_train=absent_train_bio)
print('X_train_bio-', X_train_bio.shape, ', y_train_bio-', y_train_bio.shape)



"""worldclim_species.py"""



# Load the climate grid
env_grid = np.load('climate/bioclim_h3_res4.npz')['env_grid']

cols = ['sin_lon', 'cos_lon', 'sin_lat', 'cos_lat']

env_cols = ['Annual Mean Temperature',
            'Temperature Seasonality',
            'Max Temperature of Warmest Month',
            'Min Temperature of Coldest Month',
            'Annual Precipitation',
            'Precipitation Seasonality',
            'Precipitation of Driest Quarter',
            'Elevation']

# Load the preprocessed species data
preprocessed_species = np.load('species/preprocessed_species.npz')

# Make the train df from the preprocessed species data
species_df_bio = pd.DataFrame(preprocessed_species['X_train'], columns=cols)
species_df_bio['cell'] = preprocessed_species['cell_train']
species_df_bio['id'] = preprocessed_species['id']
species_df_bio['present'] = preprocessed_species['y_train']


# climate for each cell
species_df_bio[env_cols] = env_grid[species_df_bio['cell'].values]

# Drop cells with no climate data (ocean)
clean_species = species_df_bio[env_cols].notna().all(axis=1).values
species_df_bio = species_df_bio[clean_species]

X_species_bio = species_df_bio[cols + env_cols].values

# Save
np.savez('climate/preprocessed_species_bio.npz',
         id=preprocessed_species['id'][clean_species],
         X_train=X_species_bio,
         y_train=preprocessed_species['y_train'][clean_species],
         species_train=preprocessed_species['species_train'][clean_species],
         class_id=preprocessed_species['class_id'][clean_species],
         n_sightings=preprocessed_species['n_sightings'][clean_species],
         cell_train=preprocessed_species['cell_train'][clean_species],
         grid_latlon=preprocessed_species['grid_latlon'],
         grid_cells=preprocessed_species['grid_cells'],
         res=preprocessed_species['res'],
         taxon_ids=preprocessed_species['taxon_ids'],
         env_names=np.array(env_cols))
print('X_species_bio-', X_species_bio.shape, ', dropped-', (~clean_species).sum(), 'rows')



"""worldclim_test.py"""



# Load the climate grid
env_grid = np.load('climate/bioclim_h3_res4.npz')['env_grid']

# Load the preprocessed test cells
preprocessed_test = np.load('species/preprocessed_cells_test.npz')
X_test = preprocessed_test['X_test']
y_test = preprocessed_test['y_test']
cell_test = preprocessed_test['cell_test']

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

# Make the test df from the preprocessed test cells
test_df_bio = pd.DataFrame(X_test, columns=cols)
test_df_bio['cell'] = cell_test

# climate for each cell
test_df_bio[env_cols] = env_grid[test_df_bio['cell'].values]

# Drop cells with no climate data (ocean)
clean_test = test_df_bio[env_cols].notna().all(axis=1).values
test_df_bio = test_df_bio[clean_test]

X_test_bio = test_df_bio[cols + env_cols].values
y_test_bio = y_test[clean_test]
cell_test_bio = cell_test[clean_test]

# Save
np.savez('climate/preprocessed_test_bio.npz', X_test=X_test_bio, y_test=y_test_bio, cell_test=cell_test_bio)
print('X_test_bio-', X_test_bio.shape, ', y_test_bio-', y_test_bio.shape)



"""future_bioclim.py"""



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
stats = np.load('climate/bioclim_norm_stats.npz')
hist_mu = stats['mu']
hist_sigma = stats['sigma']

# Load all selected bioclim data
future_file = 'climate/future_wc2.1_10m_bio/wc2.1_10m_bioc_MPI-ESM1-2-HR_ssp245_2041-2060.tif'

future_data_bio_1  = convert_future_tif_to_lonlat(future_file, 'Future Annual Mean Temperature', 1)
future_data_bio_4  = convert_future_tif_to_lonlat(future_file, 'Future Temperature Seasonality', 4)
future_data_bio_5  = convert_future_tif_to_lonlat(future_file, 'Future Max Temperature of Warmest Month', 5)
future_data_bio_6  = convert_future_tif_to_lonlat(future_file, 'Future Min Temperature of Coldest Month', 6)
future_data_bio_12 = convert_future_tif_to_lonlat(future_file, 'Future Annual Precipitation', 12)
future_data_bio_15 = convert_future_tif_to_lonlat(future_file, 'Future Precipitation Seasonality', 15)
future_data_bio_17 = convert_future_tif_to_lonlat(future_file, 'Future Precipitation of Driest Quarter', 17)

# use same elevation as before because it does not change
elev_data = convert_future_tif_to_lonlat('climate/wc2.1_10m_bio/wc2.1_10m_elev.tif', 'Elevation', 1)


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

np.savez('climate/future_bioclim_h3_res4.npz',
         future_env_grid=future_env_grid,
         grid_cells=np.array(grid_cells),
         future_env_names=np.array(future_col_names),
         res=res)

print(future_env_grid)

### LINK WITH THE TEST DATA #####
# Load the preprocessed test cells
preprocessed_test = np.load('species/preprocessed_cells_test.npz')
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
future_env_grid = np.load('climate/future_bioclim_h3_res4.npz')['future_env_grid']

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
np.savez('climate/preprocessed_test_future.npz', X_test=X_test_future, y_test=y_test_future, cell_test=cell_test_future)
print('X_test_future-', X_test_future.shape, ', y_test_future-', y_test_future.shape)