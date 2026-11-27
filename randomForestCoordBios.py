from global_land_mask import globe
import matplotlib.pyplot as plt
import numpy as np
import h3
import pandas as pd
import rasterio
import rasterio.features
import rasterio.warp
import seaborn as sns
from sklearn import tree
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score, roc_curve, PrecisionRecallDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize, MaxAbsScaler
import warnings
# Constants
SEED = 0
LABEL_SMOOTHING = 0.0
res=4
Abs_per_presence=1

# QoL function to print progress barr
def progressBar(part, total, length=50, sub_title=""):
    frac = part/total
    completed = int(frac * length)
    missing = length - completed
    bar = f"{sub_title}\n[{'#'* completed}{'-'*missing}]{frac:.2%}"
    return bar

# Read bio data (TODO)
preprocessed_species = np.load('preprocessed_species_bio.npz')
# Make the train df from the preprocessed species data
species_df_bio = pd.DataFrame(preprocessed_species['X_train'], columns=cols)

#Train validation split

#Stratify by species and present
stratify_crit = species_df_bio['id'].astype(str) + '_' + species_df_bio['present'].astype(str)
#species_df_bio.drop(["id","cell"], axis=1, inplace=True)
train_set, valid_set = train_test_split(species_df_bio, test_size=0.25, random_state=SEED, shuffle=True, stratify=stratify_crit)
train_set = train_set[['sin_lon', 'cos_lon', 'sin_lat', 'cos_lat', 'cell', 'present','Annual Mean Temperature', 'Temperature Seasonality','Max Temperature of Warmest Month', 'Min Temperature of Coldest Month','Annual Precipitation', 'Precipitation Seasonality','Precipitation of Driest Quarter', 'Elevation', 'id']]
valid_set = valid_set[['sin_lon', 'cos_lon', 'sin_lat', 'cos_lat', 'cell', 'present','Annual Mean Temperature', 'Temperature Seasonality','Max Temperature of Warmest Month', 'Min Temperature of Coldest Month','Annual Precipitation', 'Precipitation Seasonality','Precipitation of Driest Quarter', 'Elevation', 'id']]

# Fit Random Forest with location and climate data
rfc = RandomForestClassifier(criterion="entropy", random_state=SEED)
rfc = rfc.fit(train_set.drop(["present", "cell"],axis=1), train_set["present"])

# Calculating relevant results of model
pred_pp = rfc.predict(valid_set.drop(["present", "cell"],axis=1))
prob_pred_sp = rfc.predict_proba(train_set.drop(["present", "cell"], axis=1))
prob_val = rfc.predict_proba(valid_set.drop(["present", "cell"], axis=1))
aps = average_precision_score(train_set["present"], prob_pred_sp[:, 1])
aps_val = average_precision_score(valid_set["present"], prob_val[:, 1])
fpr, tpr, _ = roc_curve(train_set["present"], prob_pred_sp[:, 1])
mAP = np.mean(aps)
mAP_val = np.mean(aps_val)
mean_tpr = np.mean(tpr, axis=0)

# Ploting important features in Random Forest
df_fi = pd.DataFrame(columns=["feature", "importance"])
df_fi["feature"] = train_set.drop(["present", "cell"],axis=1).columns
df_fi["importance"] = rfc.feature_importances_
plt.figure(figsize=(12,5))
plt.xticks(rotation=90)
sns.barplot(df_fi, x="feature", y="importance")

# Ploting confusion matrix for validation set
matrix = confusion_matrix(valid_set["present"], pred_pp)
matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
sns.heatmap(matrix, annot=True, fmt=".2f")
plt.xticks(rotation=25)
plt.yticks(rotation=0)
plt.xlabel("Predicted labels")
plt.ylabel("True labels")
plt.show()

# tss calculation
sensitivity = matrix[1,1] / (matrix[1,1] + matrix[1,0])
specificity = matrix[0,0] / (matrix[0,0] + matrix[0,1])
tss = sensitivity + specificity - 1.0
print(tss)

# roc_auc ploting
prob_pred_sp_all = rfc.predict_proba(valid_set.drop(["present", "cell"], axis=1))[:, 1]
roc_auc = roc_auc_score(valid_set["present"], prob_pred_sp_all)
fpr, tpr, _ = roc_curve(valid_set["present"], prob_pred_sp_all)

# Result printing for train and validation
print(f"MAP_train: {mAP:.3f}\nMAP_val: {mAP_val:.3f}")
print(f"mean AUC: {np.mean(roc_auc):.3f}")

fig, ax = plt.subplots()
ax.axline((0, 0), slope=1, color="black", linestyle=(0, (5, 5)))
ax.plot(fpr, tpr, lw=2, label=f"Val AUC = {roc_auc:.3f}")
plt.xlim([-0.01,1.01])
plt.ylim([-0.01,1.01])
plt.legend()
plt.title("ROC_AUC")
plt.show()

# ======== Test ========

species_ids = train_set["id"].unique()
species_ids = np.sort(species_ids, descending=False)
X_test_bio_coords = np.load('preprocessed_test_bio_coords.npz') 
y_test_bio = np.load('preprocessed_test_bio_coords.npz')["y_test"]

ap = {}
au = {}
tpr_dic = {}
sen = {}
spe = {}
warnings.filterwarnings("ignore")
for species, n_col_species in zip(species_ids, range(0,501)):
    x_test = np.c_[X_test_bio_coords, np.full((len(X_test_bio_coords[:,1]),), species)]
    pred_test = rfc.predict(x_test)
    prob_pred_test = rfc.predict_proba(x_test)
    y_test = y_test_bio[:,n_col_species]
    matrix = confusion_matrix(y_test, pred_test)
    ap[species] = average_precision_score(y_test, prob_pred_test[:, 1])
    au[species] = roc_auc_score(y_test, prob_pred_test[:, 1])

    fpr, tpr, _ = roc_curve(y_test, prob_pred_test[:, 1])
    tpr_dic[species] = tpr
    sen[species] = matrix[1,1] / (matrix[1,1] + matrix[1,0])
    spe[species] = matrix[0,0] / (matrix[0,0] + matrix[0,1])
    print(progressBar(n_col_species, 500, sub_title=f"specie: {species}"))


mAP = np.mean(list(ap.values()))
auc = np.mean(list(au.values()))
tss = np.mean(list(sen.values())) + np.mean(list(spe.values())) - 1.0
print(f"{'='*30}\nmAP: {mAP}\nauc: {auc}\ntss: {tss}")
#mean_tpr[0], mean_tpr[-1] = 0, 1

mAP = np.mean(list(ap.values()))
auc = np.mean(list(au.values()))
tss = np.mean(list(sen.values())) + np.mean(list(spe.values())) - 1.0
print(f"{'='*30}\nmAP: {mAP}\nauc: {auc}\ntss: {tss}")