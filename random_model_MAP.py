import numpy as np
from sklearn.metrics import average_precision_score

test = np.load('species/preprocessed_cells_test.npz')
y_test, cell_test = test['y_test'], test['cell_test']
env = np.load('climate/bioclim_h3_res4.npz')['env_grid'][cell_test]
land = ~np.isnan(env).any(axis=1)

rng = np.random.default_rng(10)
ap_per_species = []
for i in range(y_test.shape[1]):
    y_score = rng.random(np.sum(land)) # 0/1 random scores
    ap_per_species.append(average_precision_score(y_test[land, i], y_score)) # avg pre per species
    
random_map = np.mean(ap_per_species)
print(f'Random model MAP: {random_map:.4f}')