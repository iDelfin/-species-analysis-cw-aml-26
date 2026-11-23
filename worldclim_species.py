import numpy as np
import pandas as pd

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