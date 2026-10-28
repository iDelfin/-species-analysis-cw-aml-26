import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
from geopy.distance import geodesic
import math

############################### Data reading ###############################

data = np.load('./Provided Code/species_train.npz')
train_locs = data['train_locs']  # 2D array, rows are number of datapoints and 
                                 # columns are "latitude" and "longitude"
train_ids = data['train_ids']    # 1D array, entries are the ID of the species 
                                 # that is present at the corresponding location in train_locs
species = data['taxon_ids']      # list of species IDe. Note these do not necessarily start at 0 (or 1)
species_names = dict(zip(data['taxon_ids'], data['taxon_names']))  # latin names of species 

# loading test data 
data_test = np.load('./Provided Code/species_test.npz', allow_pickle=True)test_locs = data_test['test_locs']    # 2D array, rows are number of datapoints 
                                      # and columns are "latitude" and "longitude"
# data_test['test_pos_inds'] is a list of lists, where each list corresponds to 
# the indices in test_locs where a given species is present, it can be assumed 
# that they are not present in the other locations 
test_pos_inds = dict(zip(data_test['taxon_ids'], data_test['test_pos_inds'])) 

# data stats
# print('Train Stats:')
# print('Number of species in train set:           ', len(species))
# print('Number of train locations:                ', train_locs.shape[0])
_, species_counts = np.unique(train_ids, return_counts=True)
# print('Average number of locations per species:  ', species_counts.mean())
# print('Minimum number of locations for a species:', species_counts.min())
# print('Maximum number of locations for a species:', species_counts.max())

############################### Create X and Y train & test ###############################

X = train_locs.copy()
y = train_ids.copy()

# To change
X_test = test_locs[test_pos_inds[31529]]
y_test = np.full(len(X_test),31529)

############################### fit model ###############################

rfc = RandomForestClassifier(criterion="entropy")
rfc = rfc.fit(X, y)

############################### acc print ###############################

acc_rf = rfc.score(X_test, y_test)
print(f"{acc_rf*100:.2f}%")