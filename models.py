import torch
import torch.nn as nn
from config import *


class MultimodalNet(nn.Module):
    """EEG encoder + face encoder -> attention-gated fusion -> classifier.
    The attention weights (alpha) say how much each modality contributed to a prediction (built-in explainability).
    A boolean `mask` (B, 2) = [eeg_available, face_available] lets the net run with a missing modality."""

    def __init__(self, n_face, n_cls=len(CLASSES), d=64, p=0.3):
        super().__init__()
        self.eeg = nn.Sequential(nn.Flatten(), nn.Linear(N_CH * N_BANDS, 128), nn.BatchNorm1d(128), nn.ELU(),
                                 nn.Dropout(p), nn.Linear(128, d), nn.ELU())
        self.face = nn.Sequential(nn.Linear(n_face, 64), nn.BatchNorm1d(64), nn.ELU(),
                                  nn.Dropout(p), nn.Linear(64, d), nn.ELU())
        self.att = nn.Sequential(nn.Linear(d, 32), nn.Tanh(), nn.Linear(32, 1))
        self.head = nn.Sequential(nn.Dropout(p), nn.Linear(d, n_cls))

    def forward(self, eeg, face, mask=None):
        h = torch.stack([self.eeg(eeg), self.face(face)], 1)          # (B, 2, d)
        s = self.att(h).squeeze(-1)                                   # (B, 2)
        if mask is not None:
            s = s.masked_fill(~mask, -1e9)
        a = s.softmax(-1)
        return self.head((a.unsqueeze(-1) * h).sum(1)), a


def fixed_mask(mods, B, dev="cpu"):
    m = {"eeg": [1, 0], "face": [0, 1], "both": [1, 1]}[mods]
    return torch.tensor(m, dtype=torch.bool, device=dev).expand(B, 2)


def random_mask(B, p, dev="cpu"):
    """Modality dropout: with prob p drop exactly one modality (robustness to sensor loss)."""
    r, which = torch.rand(B, device=dev) < p, torch.rand(B, device=dev) < .5
    m = torch.ones(B, 2, dtype=torch.bool, device=dev)
    m[r & which, 0] = False
    m[r & ~which, 1] = False
    return m


def load_model(path):
    ck = torch.load(path, map_location="cpu")
    model = MultimodalNet(ck["n_face"]); model.load_state_dict(ck["state"]); model.eval()
    return model
