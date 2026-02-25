from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GATv2Conv

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss

class AttentionLayer(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop_ratio=0., proj_drop_ratio=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn

class Attention_LastDim(nn.Module):
    def __init__(self, input_dim, hid_dim, prior_indices=None, prior_weight=0.5):
        super().__init__()
        self.attn_layer = nn.Sequential(
            nn.Linear(input_dim, hid_dim, bias=False),
            nn.ReLU(),
            nn.Linear(hid_dim, input_dim),
            nn.Sigmoid()
        )
        self.prior_indices = prior_indices
        self.prior_weight = prior_weight

    def forward(self, x):
        x_attn = self.attn_layer(x)
        if self.prior_indices:
            prior_mask = torch.zeros_like(x_attn)
            prior_mask[:, self.prior_indices] = self.prior_weight
            x_attn = x_attn + prior_mask
            x_attn = torch.clamp(x_attn, 0, 1)
        return x_attn.unsqueeze(-1)

class SingleGraphModel(nn.Module):
    def __init__(self, in_channels, hidden_dim, num_classes, heads=4, dropout=0.5):
        super().__init__()
        self.pre = nn.Linear(in_channels, hidden_dim, bias=False)
        self.gat = GATv2Conv(hidden_dim, hidden_dim // heads, heads=heads, dropout=dropout, edge_dim=1)
        self.cls = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x, edge_index=None, edge_weight=None):
        h0 = self.pre(x)
        if edge_index is not None:
            edge_attr = edge_weight.unsqueeze(-1) if edge_weight is not None else None
            h1 = F.elu(self.gat(h0, edge_index, edge_attr))
            h = h0 + h1
        else:
            h = h0
        logits = self.cls(h)
        return h, logits
