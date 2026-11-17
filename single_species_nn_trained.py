import numpy as np
import torch
from torch import nn
from pathlib import Path

# Global vars single species
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_COLS = ["sin_lon", "cos_lon", "sin_lat", "cos_lat"]
NUM_INPUTS = len(INPUT_COLS)
NUM_HIDDEN_LAYERS = 3
HIDDEN_DIM = 64
RESULTS_DIR = Path.cwd()


class SingleSpeciesMLP(nn.Module):
    def __init__(
            self, num_species=500, num_inputs=NUM_INPUTS,
            num_hidden_layers=NUM_HIDDEN_LAYERS, hidden_dim=HIDDEN_DIM, embedding_dim=32,
            activation=nn.ReLU, dropout_keep=None):
        
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

# Outputs probability that species is present at input lat lon
def species_present(model, lat, lon, sp_idx):
    model.eval()
    device = DEVICE
    lat = np.radians(lat) * 2.0
    lon = np.radians(lon)

    with torch.no_grad():
        inputs = torch.tensor([[np.sin(lon), np.cos(lon), np.sin(lat), np.cos(lat), float(sp_idx)]],
        dtype=torch.float32, device=device)
        logit = model(inputs).reshape(-1)[0]
    prob = float(torch.sigmoid(logit).item())
    
    return prob

model = SingleSpeciesMLP().to(DEVICE)
model.load_state_dict(torch.load(RESULTS_DIR / f"depth_3__width_64_weights.pt", map_location=DEVICE))
print(species_present(model, 25, -100, 414))