import csv
import json
import random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score, confusion_matrix
import matplotlib.pyplot as plt
plt.style.use('ggplot')

RESULTS = Path('ensemble_results')

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 0  # data seed- train/validation split
LEARNING_RATE = 5e-3
CRITERION = nn.BCEWithLogitsLoss(reduction='none')
BATCH_SIZE = 512
NUM_HIDDEN_LAYERS = 3
HIDDEN_DIM = 128
LABEL_SMOOTHING = 0.0
DROPOUT_KEEP = 0.85
EPOCHS = {'coords': 27, 'bio': 22, 'bio_coords': 11}  # peak validation MAP in the notebooks
EPOCHS_SINGLE = {'coords': 21, 'bio': 38, 'bio_coords': 22}
CLASS_NAMES = {0: 'Amphibia', 1: 'Aves', 2: 'Crocodylia', 3: 'Mammalia', 4: 'Squamata', 5: 'Testudines'}
METRIC_KEYS = ['ave_pre', 'roc_auc', 'tss', 'sensitivity', 'specificity']

# Settings
input_set = 'bio_coords'  # 'coords', 'bio', 'bio_coords'
NN_type = 'single_species'  # 'multi_species', 'single_species'
num_epochs = (EPOCHS if NN_type == 'multi_species' else EPOCHS_SINGLE)[input_set]
NUM_INPUTS = {'coords': 4, 'bio': 8, 'bio_coords': 12}[input_set]
seeds = [0, 1, 2]
save_predictions = True
land_only = True


# Multi species
def training_set_inputs(input_set):
    if input_set == 'coords':
        data = np.load('species/preprocessed_cells.npz')
        X_new = data['X_train']
    elif input_set == 'bio_coords':
        data = np.load('climate/preprocessed_cells_bio.npz')
        X_new = data['X_train']
    else:
        data = np.load('climate/preprocessed_cells_bio.npz')
        X_new = data['X_train'][:, 4:]  # climate columns only
    y_new = data['y_train'].astype(np.float32)
    y_nan_new = y_new.copy()
    y_nan_new[(y_nan_new == 0) & ~data['absent_train']] = np.nan
    return X_new, y_nan_new

def test_set_inputs(input_set, land_only):
    test = np.load('species/preprocessed_cells_test.npz')
    X_test, y_test, cell_test = test['X_test'], test['y_test'], test['cell_test']
    rows = np.arange(len(X_test))  # position
    if input_set == 'coords' and not land_only:
        return X_test, y_test, rows

    env_grid = np.load('climate/bioclim_h3_res4.npz')['env_grid']
    env_test = env_grid[cell_test]
    land = ~np.isnan(env_test).any(axis=1)  # cells with climate
    if input_set == 'coords':
        X = X_test
    elif input_set == 'bio':
        X = env_test
    else:
        X = np.column_stack([X_test, env_test])
    return X[land], y_test[land], rows[land]

def inputs_targets(which_X, which_y, which_mask):
    inputs = torch.from_numpy(which_X).to(torch.float32).to(DEVICE)
    targets = torch.from_numpy(which_y).to(torch.float32).to(DEVICE)
    mask = torch.from_numpy(which_mask).to(torch.float32).to(DEVICE)
    return TensorDataset(inputs, targets, mask)

def make_data_loaders(X_train, y_train, train_mask, X_valid, y_valid, valid_mask, batch_size=BATCH_SIZE, seed=SEED):
    train_gen = torch.Generator().manual_seed(seed)
    train_dataloader = DataLoader(inputs_targets(X_train, y_train, train_mask), batch_size=batch_size,
                                  shuffle=True, generator=train_gen)
    valid_dataloader = DataLoader(inputs_targets(X_valid, y_valid, valid_mask), batch_size=batch_size,
                                  shuffle=False, generator=train_gen)
    return train_dataloader, valid_dataloader

class MultiSpeciesMLP(nn.Module):
    def __init__(
            self, num_species=500, num_inputs=NUM_INPUTS,
            num_hidden_layers=NUM_HIDDEN_LAYERS, hidden_dim=HIDDEN_DIM,
            activation=nn.ReLU, dropout_keep=DROPOUT_KEEP):
        
        super().__init__()
        self.num_inputs = num_inputs
        self.num_hidden_layers = num_hidden_layers
        self.hidden_dim = hidden_dim
        
        layers = []
        input_dim = num_inputs
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(input_dim, hidden_dim))
            if dropout_keep is not None:
                layers.append(nn.Dropout(p=1.0 - dropout_keep))
            layers.append(activation())
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, num_species))
        self.model = nn.Sequential(*layers)
        for layer in self.model.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, inputs):
        return self.model(inputs)

def masked_loss(logits, targets, mask, criterion=CRITERION):
    loss = criterion(logits, targets)
    return (loss * mask).sum() / mask.sum()

def evaluate(model, data_loader, criterion=CRITERION, threshold=0.5):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    logits_all, targets_all, mask_all = [], [], []

    with torch.no_grad():
        for inputs, targets, mask in data_loader:
            logits = model(inputs)
            total_loss += (criterion(logits, targets) * mask).sum().item()
            predict = (torch.sigmoid(logits) >= threshold).to(targets.dtype)
            total_correct += ((predict == targets) * mask).sum().item()
            total_examples += mask.sum().item()
            logits_all.append(logits)
            targets_all.append(targets)
            mask_all.append(mask)

    probabilities = torch.sigmoid(torch.cat(logits_all)).cpu().numpy()
    true_out = torch.cat(targets_all).cpu().numpy()
    mask_out = torch.cat(mask_all).cpu().numpy().astype(bool)

    auc_per_species = np.full(true_out.shape[1], np.nan)
    ap_per_species = np.full(true_out.shape[1], np.nan)
    for species in range(true_out.shape[1]):
        mask = mask_out[:, species]
        targs = true_out[mask, species]
        probs = probabilities[mask, species]
        if len(np.unique(targs)) < 2:
            continue
        auc_per_species[species] = roc_auc_score(targs, probs)
        ap_per_species[species] = average_precision_score(targs, probs)

    curve_data = {'macro_auc': float(np.nanmean(auc_per_species)),
                  'macro_map': float(np.nanmean(ap_per_species))}
    return total_loss / total_examples, total_correct / total_examples, curve_data

def train_model(model, train_data, valid_data, num_epochs, learning_rate=LEARNING_RATE):
    optimiser = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=3e-4)
    stats = []

    def record():
        train_loss, train_acc, train_curves = evaluate(model, train_data)
        valid_loss, valid_acc, valid_curves = evaluate(model, valid_data)
        stats.append([train_loss, valid_loss, train_acc, valid_acc, train_curves['macro_auc'], valid_curves['macro_auc'],
                      train_curves['macro_map'], valid_curves['macro_map']])

    record()
    for epoch in range(1, num_epochs + 1):
        model.train()
        for inputs, targets, mask in train_data:
            optimiser.zero_grad()
            loss = masked_loss(model(inputs), targets, mask)
            loss.backward()
            optimiser.step()
        record()
        print(f'  epoch {epoch:3d}/{num_epochs}: valid MAP {stats[-1][7]:.4f}', flush=True)
    return np.asarray(stats)

def name_builder(num_inputs, num_epochs, seed):
    '''Run name creation for naming of files with associated settings'''
    prefix = 'multi_sp' if NN_type == 'multi_species' else 'single_sp'
    return '__'.join([f'{prefix}_depth_{NUM_HIDDEN_LAYERS}', f'width_{HIDDEN_DIM}', f'inputs_{num_inputs}',
                      f'epochs_{num_epochs}', f'dropout_{DROPOUT_KEEP}', f'seed_{seed}'])

def reset_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def multi_species_predict(model, input, batch_size=1024):
    model.eval()
    device = DEVICE
    prob = []

    with torch.no_grad():
        for i in range(0, len(input), batch_size):
            input_tens = torch.tensor(input[i : i+batch_size], dtype=torch.float32, device=device)
            prob.append(torch.sigmoid(model(input_tens)).cpu().detach().numpy())

    return np.vstack(prob)

def optimise_thresholds_multi(probs_valid, y_valid, valid_mask):
    thresholds = np.full(y_valid.shape[1], 0.5)
    for species in range(y_valid.shape[1]):
        mask = valid_mask[:, species].astype(bool)
        targs = y_valid[mask, species]
        probs = probs_valid[mask, species]
        if (targs == 1).sum() < 3 or (targs == 0).sum() < 3:
            continue
        false_pos_rate, true_pos_rate, threshold = roc_curve(targs, probs)
        true_skill_stat = true_pos_rate - false_pos_rate
        thresh_idx = int(np.argmax(true_skill_stat))
        if true_skill_stat[thresh_idx] > 0 and np.isfinite(threshold[thresh_idx]):
            thresholds[species] = threshold[thresh_idx]
    return thresholds

def per_species_metrics_multi(probs, targets, thresholds, mask=None):
    metrics = {key: np.full(targets.shape[1], np.nan) for key in METRIC_KEYS}

    for species in range(targets.shape[1]):
        if mask is not None:
            mask_m = mask[:, species].astype(bool)
            targs = targets[mask_m, species].astype(int)
            probs_p = probs[mask_m, species]
        else:
            targs = targets[:, species].astype(int)
            probs_p = probs[:, species]
        if len(np.unique(targs)) < 2:
            continue

        predicts = (probs_p >= thresholds[species]).astype(int)
        tru_pos, fal_neg, fal_pos, tru_neg = confusion_matrix(targs, predicts, labels=[1, 0]).ravel()
        sensitivity = tru_pos / (tru_pos + fal_neg)
        specificity = tru_neg / (tru_neg + fal_pos)

        metrics['roc_auc'][species] = roc_auc_score(targs, probs_p)
        metrics['ave_pre'][species] = average_precision_score(targs, probs_p)
        metrics['tss'][species] = sensitivity + specificity - 1.0
        metrics['sensitivity'][species] = sensitivity
        metrics['specificity'][species] = specificity
    return metrics

def macro(metrics):
    return {key: float(np.nanmean(metrics[key])) for key in METRIC_KEYS}

def per_class(metrics, class_id):
    rows = []
    for cl, class_name in CLASS_NAMES.items():
        in_class = class_id == cl
        row = {'tax_class': class_name, 'species_count': int(in_class.sum())}
        for key in METRIC_KEYS:
            row[key] = float(np.nanmean(metrics[key][in_class])) if in_class.any() else np.nan
        rows.append(row)
    return rows


# Single species
def training_set_inputs_single(input_set):
    data = np.load('climate/preprocessed_species_bio.npz')  # rows with climate
    if input_set == 'coords':
        X_new = data['X_train'][:, :4]
    elif input_set == 'bio':
        X_new = data['X_train'][:, 4:]
    else:
        X_new = data['X_train']
    return X_new, data['id'], data['y_train'].astype(np.float32)

def inputs_targets_single(which_X, which_ids, which_y):
    inputs = torch.from_numpy(np.column_stack([which_X, which_ids])).to(torch.float32).to(DEVICE)  # species index last
    targets = torch.from_numpy(which_y).to(torch.float32).to(DEVICE)
    return TensorDataset(inputs, targets)

def make_data_loaders_single(X_train, ids_train, y_train, batch_size=BATCH_SIZE, seed=SEED):
    train_gen = torch.Generator().manual_seed(seed)
    return DataLoader(inputs_targets_single(X_train, ids_train, y_train), batch_size=batch_size,
                      shuffle=True, generator=train_gen)

class SingleSpeciesMLP(nn.Module):
    def __init__(
            self, num_species=500, num_inputs=NUM_INPUTS,
            num_hidden_layers=NUM_HIDDEN_LAYERS, hidden_dim=HIDDEN_DIM, embedding_dim=32,
            activation=nn.ReLU, dropout_keep=DROPOUT_KEEP):
        
        super().__init__()
        self.num_inputs = num_inputs
        self.num_hidden_layers = num_hidden_layers
        self.hidden_dim = hidden_dim
        self.species_embed = nn.Embedding(num_species, embedding_dim)

        layers = []
        input_dim = num_inputs + embedding_dim
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(input_dim, hidden_dim))
            if dropout_keep is not None:
                layers.append(nn.Dropout(p=1.0 - dropout_keep))
            layers.append(activation())
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, 1))
        self.model = nn.Sequential(*layers)
        for layer in self.model.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, inputs):
        input_feats = inputs[:, :self.num_inputs]
        species_idx = inputs[:, self.num_inputs].to(torch.long)
        species_emb = self.species_embed(species_idx)
        inputs_emb = torch.cat([input_feats, species_emb], dim=-1)
        return self.model(inputs_emb).squeeze(-1)

def train_model_single(model, train_data, num_epochs, learning_rate=LEARNING_RATE):
    optimiser = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    stats = []
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss, total_rows = 0.0, 0
        for inputs, targets in train_data:
            optimiser.zero_grad()
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimiser.step()
            total_loss += loss.item() * len(targets)
            total_rows += len(targets)
        stats.append(total_loss / total_rows)
        print(f'  epoch {epoch:3d}/{num_epochs}: train loss {stats[-1]:.4f}', flush=True)
    return np.asarray(stats)

def single_species_predict_valid(model, input, batch_size=1024):
    model.eval()
    device=DEVICE
    examples = len(input)
    probs = []

    with torch.no_grad():
        for i in range(0, examples, batch_size):
                input_tens = torch.as_tensor(input[i : i+batch_size], dtype=torch.float32, device=device)
                probs.append(torch.sigmoid(model(input_tens)).cpu().detach().numpy())
            

    return np.concatenate(probs)

def single_species_predict_test(model, input, batch_size=1024):
    model.eval()
    device = DEVICE
    examples = len(input)
    probs = np.zeros((examples, 500))
    input_tens = torch.as_tensor(input, dtype=torch.float32, device=DEVICE)
     
    with torch.no_grad():
        for species in range(500):
            species_col = torch.full((examples, 1), species, dtype=torch.float32, device=DEVICE)
            species_in = torch.cat([input_tens, species_col], dim=1)
            species_prob = []
            for i in range(0, examples, batch_size):
                species_prob.append(torch.sigmoid(model(species_in[i : i + batch_size])).cpu().numpy().reshape(-1))
            probs[:, species] = np.concatenate(species_prob)
    return probs

def rows_to_matrix(values, ids, num_species=500):
    matrix = np.zeros((len(values), num_species), dtype=np.float32)
    mask = np.zeros((len(values), num_species), dtype=np.float32)
    matrix[np.arange(len(values)), ids] = values
    mask[np.arange(len(values)), ids] = 1.0
    return matrix, mask


# Plots
def plot_macro(table, path):
    labels = [row[0] for row in table]
    shown = {'ave_pre': 'MAP', 'tss': 'TSS', 'sensitivity': 'Sensitivity', 'specificity': 'Specificity'}
    x = np.arange(len(labels))
    width = 0.8 / len(shown)
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (key, name) in enumerate(shown.items()):
        ax.bar(x + i * width, [row[1][key] for row in table], width, label=name)
    ax.set_xticks(x + width * (len(shown) - 1) / 2)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel('Score')
    ax.set_title(f'Macro test metrics: {NN_type}, {input_set}')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)

def plot_per_class(class_rows, path):
    models = list(dict.fromkeys(row['model'] for row in class_rows))
    classes = list(CLASS_NAMES.values())
    x = np.arange(len(classes))
    width = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, model_name in enumerate(models):
        values = [next(r['ave_pre'] for r in class_rows if r['model'] == model_name and r['tax_class'] == c) for c in classes]
        ax.bar(x + i * width, values, width, label=model_name)
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(classes)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Taxonomic class')
    ax.set_ylabel('Macro average precision')
    ax.set_title(f'Test MAP by taxonomic class: {NN_type}, {input_set}')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)

# Training and evaluation
RESULTS.mkdir(exist_ok=True)
class_id = np.load('species/preprocessed_cells.npz')['class_id']  # class of species 0-499

if NN_type == 'single_species':
    X, ids, y = training_set_inputs_single(input_set)
    num_inputs = X.shape[1]
    print(f'{NN_type}, input set: {input_set} ({num_inputs} inputs), epochs: {num_epochs}, seeds: {seeds}, device: {DEVICE}')
    stratify_criteria = ids.astype(str) + '_' + y.astype(int).astype(str)
    X_train, X_valid, ids_train, ids_valid, y_train, y_valid_rows = train_test_split(
        X, ids, y, test_size=0.25, random_state=SEED, shuffle=True, stratify=stratify_criteria)
    y_valid, valid_mask = rows_to_matrix(y_valid_rows, ids_valid)
else:
    X, y_nan = training_set_inputs(input_set)
    num_inputs = X.shape[1]
    print(f'{NN_type}, input set: {input_set} ({num_inputs} inputs), epochs: {num_epochs}, seeds: {seeds}, device: {DEVICE}')

    # Stratify criteria - 0 species present (pseudo absences), 1, 2, 3, 4, or >5 species present (6 categories to stratify on)
    # Ensures equal presences and absences in each set.
    stratify_criteria = np.clip((y_nan == 1).sum(axis=1), 0, 5)
    X_train, X_valid, y_train, y_valid = train_test_split(X, y_nan, test_size=0.25, random_state=SEED, shuffle=True, stratify=stratify_criteria)

    train_mask = y_train.copy()
    train_mask[train_mask == 0.0] = 1.0
    train_mask[np.isnan(train_mask)] = 0.0
    valid_mask = y_valid.copy()
    valid_mask[valid_mask == 0.0] = 1.0
    valid_mask[np.isnan(valid_mask)] = 0.0

    y_train[y_train == 0.0] = LABEL_SMOOTHING
    y_valid[y_valid == 0.0] = LABEL_SMOOTHING

    X_train[np.isnan(X_train)] = 0.0
    X_valid[np.isnan(X_valid)] = 0.0
    y_train[np.isnan(y_train)] = 0.0
    y_valid[np.isnan(y_valid)] = 0.0

X_test, y_test, test_rows = test_set_inputs(input_set, land_only)
X_test = np.nan_to_num(X_test).astype(np.float32)
print(f'Train {X_train.shape}, valid {X_valid.shape}, test {X_test.shape}')

table = []
probs_valid_all = []
probs_test_sum = np.zeros(y_test.shape, dtype=np.float64)
probs_test_sumsq = np.zeros(y_test.shape, dtype=np.float64)

for seed in seeds:
    name = name_builder(num_inputs, num_epochs, seed)
    weights_path = RESULTS / f'{name}_weights.pt'
    reset_seeds(seed)  # weight init, dropout and batch order change with the seed
    if NN_type == 'single_species':
        model = SingleSpeciesMLP(num_inputs=num_inputs).to(DEVICE)
    else:
        model = MultiSpeciesMLP(num_inputs=num_inputs).to(DEVICE)

    if weights_path.exists():
        model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
        print(f'Loaded {name}')
    else:
        print(f'Training {name} on {DEVICE}', flush=True)
        if NN_type == 'single_species':
            train_loader = make_data_loaders_single(X_train, ids_train, y_train, seed=seed)
            stats = train_model_single(model, train_loader, num_epochs)
        else:
            train_loader, valid_loader = make_data_loaders(X_train, y_train, train_mask, X_valid, y_valid, valid_mask, seed=seed)
            stats = train_model(model, train_loader, valid_loader, num_epochs)
        torch.save(model.state_dict(), weights_path)
        config = dict(seed=seed, data_seed=SEED, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE, num_epochs=num_epochs, num_hidden_layers=NUM_HIDDEN_LAYERS, hidden_dim=HIDDEN_DIM, dropout_keep=DROPOUT_KEEP, input_set=input_set, nn_type=NN_type)
        np.savez(RESULTS / f'{name}.npz', stats=stats, config=json.dumps(config, sort_keys=True))

    if NN_type == 'single_species':
        probs_valid, _ = rows_to_matrix(single_species_predict_valid(model, np.column_stack([X_valid, ids_valid])), ids_valid)
    else:
        probs_valid = multi_species_predict(model, X_valid)
    probs_valid_all.append(probs_valid)
    thresholds = optimise_thresholds_multi(probs_valid, y_valid, valid_mask)
    valid_metrics = macro(per_species_metrics_multi(probs_valid, y_valid, thresholds, mask=valid_mask))

    if NN_type == 'single_species':
        probs_test = single_species_predict_test(model, X_test)
    else:
        probs_test = multi_species_predict(model, X_test)
    test_species = per_species_metrics_multi(probs_test, y_test, thresholds)
    probs_test_sum += probs_test
    probs_test_sumsq += probs_test.astype(np.float64) ** 2
    del probs_test

    test_metrics = macro(test_species)
    table.append([f'seed_{seed}', valid_metrics, test_metrics, test_species])
    print(f'  seed {seed}: valid MAP {valid_metrics["ave_pre"]:.4f} | test MAP {test_metrics["ave_pre"]:.4f}, '
            f'AUC {test_metrics["roc_auc"]:.4f}, TSS {test_metrics["tss"]:.4f}', flush=True)

# Ensemble- average prob of seeds
n = len(seeds)
probs_valid_ens = np.mean(probs_valid_all, axis=0)
thresholds_ens = optimise_thresholds_multi(probs_valid_ens, y_valid, valid_mask)
valid_metrics = macro(per_species_metrics_multi(probs_valid_ens, y_valid, thresholds_ens, mask=valid_mask))
probs_test_ens = (probs_test_sum / n).astype(np.float32)
probs_test_std = np.sqrt(np.maximum(probs_test_sumsq / n - (probs_test_sum / n) ** 2, 0)).astype(np.float32)  # disagreeement
test_species = per_species_metrics_multi(probs_test_ens, y_test, thresholds_ens)
table.append([f'ensemble_{n}_seeds', valid_metrics, macro(test_species), test_species])

# Macro summary
print(f'\n{"model":>18} | valid MAP | valid AUC | test MAP | test AUC | test TSS | test sens | test spec')
summary = []
for label, v, t, _ in table:
    row = [label, v['ave_pre'], v['roc_auc']] + [t[k] for k in METRIC_KEYS]
    summary.append(row)
    print(f'{label:>18} | {row[1]:9.4f} | {row[2]:9.4f} | {row[3]:8.4f} | {row[4]:8.4f} | {row[5]:8.4f} | {row[6]:9.4f} | {row[7]:9.4f}')
print(f'Mean seed-to-seed std of test probabilities: {probs_test_std.mean():.4f}')

tag = f'{NN_type}_{input_set}{"_land" if land_only else ""}_{n}seeds'
with (RESULTS / f'summary_nn_{tag}.csv').open('w', newline='', encoding='utf-8') as handle:
    writer = csv.writer(handle)
    writer.writerow(['model', 'valid_map', 'valid_auc'] + [f'test_{k}' for k in METRIC_KEYS])
    writer.writerows(summary)

# Per taxonomic class
class_rows = []
for label, _, _, species_metrics in table:
    for row in per_class(species_metrics, class_id):
        class_rows.append({'model': label, **row})
with (RESULTS / f'per_class_nn_{tag}.csv').open('w', newline='', encoding='utf-8') as handle:
    writer = csv.DictWriter(handle, fieldnames=['model', 'tax_class', 'species_count'] + METRIC_KEYS)
    writer.writeheader()
    writer.writerows(class_rows)
ens_label = table[-1][0]
print('Ensemble test MAP by class: ' + ', '.join(f'{r["tax_class"]} {r["ave_pre"]:.3f}' for r in class_rows if r['model'] == ens_label))

plot_macro([[row[0], row[2]] for row in table], RESULTS / f'macro_metrics_{tag}.png')
plot_per_class(class_rows, RESULTS / f'per_class_map_{tag}.png')


if save_predictions:
    pres_abs = probs_test_ens >= thresholds_ens
    np.savez(RESULTS / f'predictions_nn_{tag}.npz', probs_mean=probs_test_ens.astype(np.float16), probs_std=probs_test_std.astype(np.float16), pres_abs=pres_abs, thresholds=thresholds_ens, test_rows=test_rows, seeds=seeds)