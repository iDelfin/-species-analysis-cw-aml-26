import numpy as np
import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor

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
df.to_csv('species_taxonomy.csv', index=False)
print(df['class'].value_counts(dropna=False))
