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