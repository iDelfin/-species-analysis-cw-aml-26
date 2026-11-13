import numpy as np
import pandas as pd
from global_land_mask import globe
import h3

SEED=0
res=4
Background_per_cell=1

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

# Remove sightings- about 25km from land
df = df[near_land(df['lat'].values, df['lon'].values)].copy()
print('Total after removing ocean sightings-', len(df))

# Bin- find the H3 cell each sighting falls in
df['cell'] = [cell_index[h3.latlng_to_cell(lat, lon, res)] for lat, lon in zip(df['lat'], df['lon'])]

# Index for species 0-499
species_index = {taxon: i for i, taxon in enumerate(train['taxon_ids'])}
df['index'] = df['species'].map(species_index)

# Presence only- one row per cell, list of species seen there
presences = df.groupby('cell')['index'].unique().apply(sorted).reset_index(name='species')
presences['n_sightings'] = df.groupby('cell').size().values
print('Cells with sightings-', len(presences))

# Pseudo-absences- random empty cells (no species seen), empty species list
rng = np.random.default_rng(SEED)
candidates = np.setdiff1d(np.arange(len(grid)), presences['cell']) # cells where no species was seen
picked = rng.choice(candidates, len(presences)*Background_per_cell, replace=False)
absences = pd.DataFrame({'cell': picked, 'species': [[] for _ in picked], 'n_sightings': 0})

data = pd.concat([presences, absences], ignore_index=True)
data['n_species'] = data['species'].apply(len)
print('Background cells-', len(absences))

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

# Save
np.savez('species/preprocessed_cells.npz', X_train=X_train, y_train=y_train, class_id=class_id, n_sightings=data['n_sightings'].values, n_species=data['n_species'].values, cell_train=data['cell'].values, grid_latlon=grid, grid_cells=np.array(grid_cells), res=res, taxon_ids=train['taxon_ids'])
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

print(train_df.head())