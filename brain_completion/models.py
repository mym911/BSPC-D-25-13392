from __future__ import annotations
import torch
import torch.nn as nn

class ROICompletionGenerator(nn.Module):
    def __init__(self, cond_dim: int, num_rois: int):
        super().__init__()
        self.cond_embed = nn.Linear(cond_dim, num_rois)
        self.encoder = nn.Sequential(
            nn.Conv1d(num_rois, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
        )
        self.decoder = nn.Sequential(
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, num_rois, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor, mask: torch.Tensor):
        cond_emb = self.cond_embed(cond).unsqueeze(-1).expand(-1, -1, x.shape[-1])
        x = x * mask.float()
        x = x + cond_emb * mask.float()
        return self.decoder(self.encoder(x))

class ROICompletionDiscriminator(nn.Module):
    def __init__(self, num_rois: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(num_rois, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(64, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        x = x * mask.float()
        feat = self.pool(self.conv(x)).view(x.size(0), -1)
        return self.fc(feat)
