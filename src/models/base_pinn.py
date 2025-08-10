# src/models/base_pinn.py

import torch
import torch.nn as nn

class BasePINN(nn.Module):
    def __init__(self, input_dim=2, output_dim=1, hidden_layers=[50, 50, 50], activation=nn.Tanh):
        super(BasePINN, self).__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_layers:
            layers.append(nn.Linear(in_dim, h))
            layers.append(activation())
            in_dim = h
        layers.append(nn.Linear(in_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
