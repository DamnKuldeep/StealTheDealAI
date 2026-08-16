import json
import logging
import torch.nn as nn
from sklearn.feature_extraction.text import HashingVectorizer
import numpy as np

from config import settings

# Loaded from models/dnn_norm_stats.json, written by notebooks/02_train_neural_network.ipynb
# after training on Kaggle. Falls back to placeholders (with a warning) if training hasn't
# happened yet - predictions will be numerically wrong until the real file is in place.
Y_MEAN = 7.5  # placeholder
Y_STD = 1.2   # placeholder

if settings.DNN_NORM_STATS_PATH.exists():
    with open(settings.DNN_NORM_STATS_PATH) as f:
        _stats = json.load(f)
    Y_MEAN = _stats["y_mean"]
    Y_STD = _stats["y_std"]
else:
    logging.warning(
        f"{settings.DNN_NORM_STATS_PATH} not found - using placeholder Y_MEAN/Y_STD. "
        "DNN price predictions will be wrong until you run notebooks/02_train_neural_network.ipynb "
        "on Kaggle and copy deep_neural_network.pth + dnn_norm_stats.json into models/."
    )

class ResidualBlock(nn.Module):
    def __init__(self, hidden_size, dropout_prob):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        return self.relu(out)

class DeepNeuralNetwork(nn.Module):
    def __init__(self, input_size=5000, num_layers=10, hidden_size=2048, dropout_prob=0.2):
        super(DeepNeuralNetwork, self).__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
        )
        self.residual_blocks = nn.ModuleList()
        for i in range(num_layers - 2):
            self.residual_blocks.append(ResidualBlock(hidden_size, dropout_prob))
        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.input_layer(x)
        for block in self.residual_blocks:
            x = block(x)
        return self.output_layer(x)

def get_vectorizer():
    return HashingVectorizer(n_features=5000, stop_words="english", binary=True)

def inverse_transform_price(y_scaled):
    y_log = (y_scaled * Y_STD) + Y_MEAN
    return np.expm1(y_log)
