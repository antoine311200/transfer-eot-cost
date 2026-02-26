import math

import torch
import torch.nn as nn


class node_wrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, t, x, *args, **kwargs):
        t = t.to(x.device)
        if t.dim() < 2:
            t = t.repeat(x.shape[0])[:, None]
        return self.model(torch.cat([x, t], dim=1))


class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=t.device) * -embeddings)
        embeddings = t * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ResBlock(nn.Module):
    def __init__(self, dim, time_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
        self.time_mlp = nn.Linear(time_dim, dim)
        self.act = nn.SiLU()

    def forward(self, x, t_emb):
        if t_emb is None:
            time_shift = 0
        else:
            time_shift = self.time_mlp(t_emb)
        h = self.mlp(x + time_shift)
        return x + h


class MLP2(nn.Module):
    def __init__(self, dim=3, out_dim=3, w=256, time_varying=False, time_dim=128):
        super().__init__()

        hidden_dim = w

        self.time_varying = time_varying
        if self.time_varying:
            self.t_embed = nn.Sequential(
                TimeEmbedding(time_dim),
                nn.Linear(time_dim, time_dim),
                nn.SiLU()
            )
        self.input_proj = nn.Linear(dim, hidden_dim)
        self.layers = nn.ModuleList([
            ResBlock(hidden_dim, time_dim),
            ResBlock(hidden_dim, time_dim),
            ResBlock(hidden_dim, time_dim)
        ])
        self.out = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, t, x):
        if self.time_varying:
            if t.dim() == 1: t = t.unsqueeze(-1)
            t_emb = self.t_embed(t)
        else:
            t_emb = None
        h = self.input_proj(x)
        for layer in self.layers: h = layer(h, t_emb)
        return self.out(h)


class MLP(torch.nn.Module):
    def __init__(self, dim, out_dim=None, w=64, n_layers=2, time_varying=False, use_norm=None):
        super().__init__()
        self.time_varying = time_varying
        if out_dim is None:
            out_dim = dim

        layers = []
        layers.append(torch.nn.Linear(dim + (1 if time_varying else 0), w))
        if use_norm == "batch":
            layers.append(torch.nn.BatchNorm1d(w))
        elif use_norm == "layer":
            layers.append(torch.nn.LayerNorm(w))
        layers.append(torch.nn.SELU())
        for _ in range(n_layers - 1):
            layers.append(torch.nn.Linear(w, w))
            if use_norm == "batch":
                layers.append(torch.nn.BatchNorm1d(w))
            elif use_norm == "layer":
                layers.append(torch.nn.LayerNorm(w))
            layers.append(torch.nn.SELU())
        layers.append(torch.nn.Linear(w, out_dim))
        self.net = torch.nn.Sequential(*layers)

        # self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # LeCun Normal is preferred for SELU
                # std = 1 / sqrt(fan_in)
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='linear')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Special handling for the final layer in Flow Matching:
        # Initializing the output layer with a smaller scale helps 
        # prevent wild trajectories early in training.
        final_layer = self.net[-1]
        if isinstance(final_layer, nn.Linear):
            nn.init.xavier_uniform_(final_layer.weight, gain=0.01)
            if final_layer.bias is not None:
                nn.init.zeros_(final_layer.bias)

    def forward(self, x):
        return self.net(x)
