import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from global_land_mask import globe

SEED=0
Abs_per_presence=1

# Load data
train = np.load('./Provided Code/species_train.npz')
test = np.load('./Provided Code/species_test.npz', allow_pickle=True)
grid = test['test_locs'] # grid points (lat, lon)

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
df = df[near_land(df['lat'].values, df['lon'].values)]
print('Total after removing ocean sightings-', len(df))

# Bin- move every sighting to its nearest grid point
nn = NearestNeighbors(n_neighbors=1, metric='haversine').fit(np.radians(grid))
df['cell'] = nn.kneighbors(np.radians(df[['lat', 'lon']].values), return_distance=False)[:, 0]

# Presence only- one row per, dups removed
presences = df[['cell', 'species']].drop_duplicates()
presences['present'] = 1
print('Presences after binning-', len(presences))

# Pseudo-absences- random grid cells
rng = np.random.default_rng(SEED)
absences = []
for species, group in presences.groupby('species'):
    candidates = np.setdiff1d(np.arange(len(grid)), group['cell'])
    cells = rng.choice(candidates, len(group)*Abs_per_presence, replace=False)
    absences.append(pd.DataFrame({'cell': cells, 'species': species, 'present': 0}))

data = pd.concat([presences]+absences, ignore_index=True)
print('Pseudo-absences-', (data['present']==0).sum())

# Features- normalise lat/lon to [-1,1], then sin/cos (SINR paper approach)
def encode(latlon):
    lat = latlon[:,0]/90
    lon = latlon[:,1]/180
    return np.column_stack([np.sin(np.pi*lon), np.cos(np.pi*lon), np.sin(np.pi*lat), np.cos(np.pi*lat)])

X_train = encode(grid[data['cell']])
y_train = data['present'].values
species_train = data['species'].values
X_test = encode(grid)

# Save
np.savez('./PreprocessData/preprocessed_data.npz', X_train=X_train, y_train=y_train, species_train=species_train, X_test=X_test, cell_train=data['cell'].values, taxon_ids=train['taxon_ids'])
print('X_train-', X_train.shape, ', y_train-', y_train.shape, ', X_test-', X_test.shape)

preprocessed = np.load('./PreprocessData/preprocessed_data.npz')
cols = ['sin_lon', 'cos_lon', 'sin_lat', 'cos_lat']
train_df = pd.DataFrame(preprocessed['X_train'], columns=cols)
train_df['species'] = preprocessed['species_train']
train_df['present'] = preprocessed['y_train']
test_df = pd.DataFrame(preprocessed['X_test'], columns=cols)

# For model inputs
X = train_df[cols].values
y = train_df['present'].values

print(X,y)
print(train_df.head())
