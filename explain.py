"""Explainability: Integrated Gradients (Captum) on BOTH modalities + attention-based modality weights.
Local  : why did the model predict this state for THIS window? (channel x band map, face-feature bars, modality weights)
Global : which brain regions/bands/facial cues matter on average?"""
import argparse
import numpy as np, torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from captum.attr import IntegratedGradients
from config import *
from models import load_model
import face_features as ff

BAND_NAMES = list(BANDS)


class _Wrap(nn.Module):
    def __init__(self, m, mask):
        super().__init__(); self.m = m; self.register_buffer("mask", mask)

    def forward(self, e, f):
        return self.m(e, f, self.mask.expand(e.shape[0], -1))[0]


def explain(model, e, f, mask=(True, True), target=None, n_steps=32):
    """e: (32,4) tensor, f: (N_FACE,) tensor (both z-scored). Baseline = 0 = the person's own average."""
    m = torch.tensor(mask, dtype=torch.bool)
    e, f = e.float().unsqueeze(0), f.float().unsqueeze(0)
    model.eval()
    with torch.no_grad():
        logits, alpha = model(e, f, m.expand(1, -1))
    probs = logits.softmax(-1)[0]
    t = int(probs.argmax()) if target is None else int(target)
    ae, af = IntegratedGradients(_Wrap(model, m)).attribute(
        (e, f), baselines=(torch.zeros_like(e), torch.zeros_like(f)), target=t, n_steps=n_steps)
    return dict(probs=probs.numpy(), target=t, alpha=alpha[0].numpy(),
                eeg_attr=ae[0].detach().numpy(), face_attr=af[0].detach().numpy())


def describe(res, f_vals, k=3):
    """Short human-readable explanation (also used as the live overlay)."""
    ae, af = res["eeg_attr"], res["face_attr"]
    lines = [f"{CLASSES[res['target']]} {res['probs'][res['target']]:.0%} | EEG {res['alpha'][0]:.0%} Face {res['alpha'][1]:.0%}"]
    eeg = []
    for i in np.argsort(-np.abs(ae).ravel())[:k]:
        c, b = np.unravel_index(i, ae.shape)
        eeg.append(f"{BAND_NAMES[b]}@{CH_NAMES[c]}{'+' if ae[c, b] > 0 else '-'}")
    face = [f"{ff.FACE_FEATURE_NAMES[i]} {'high' if f_vals[i] > 0 else 'low'}{'+' if af[i] > 0 else '-'}"
            for i in np.argsort(-np.abs(af))[:k]]
    return lines + ["EEG: " + ", ".join(eeg), "Face: " + ", ".join(face), "(+ supports / - opposes prediction)"]


def plot_local(res, f_vals, path):
    fig, ax = plt.subplots(1, 3, figsize=(17, 4), gridspec_kw={"width_ratios": [3, 2, 1]})
    ae = res["eeg_attr"]; v = np.abs(ae).max() + 1e-9
    im = ax[0].imshow(ae.T, cmap="coolwarm", vmin=-v, vmax=v, aspect="auto")
    ax[0].set_xticks(range(N_CH)); ax[0].set_xticklabels(CH_NAMES, rotation=90, fontsize=7)
    ax[0].set_yticks(range(N_BANDS)); ax[0].set_yticklabels(BAND_NAMES)
    ax[0].set_title("EEG attribution (channel x band)"); plt.colorbar(im, ax=ax[0])
    top = np.argsort(-np.abs(res["face_attr"]))[:8][::-1]
    ax[1].barh([ff.FACE_FEATURE_NAMES[i] for i in top], res["face_attr"][top],
               color=["tab:red" if x > 0 else "tab:blue" for x in res["face_attr"][top]])
    ax[1].set_title("Face attribution (top 8)")
    ax[2].bar(["EEG", "Face"], res["alpha"], color=["tab:purple", "tab:orange"]); ax[2].set_ylim(0, 1)
    ax[2].set_title("Modality weight")
    fig.suptitle(describe(res, f_vals)[0]); fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def global_importance(model, E, F, n=200, seed=0):
    idx = np.random.default_rng(seed).choice(len(E), min(n, len(E)), replace=False)
    ae, af, al = 0, 0, 0
    for i in idx:
        r = explain(model, torch.tensor(E[i]), torch.tensor(F[i]), n_steps=16)
        ae = ae + np.abs(r["eeg_attr"]); af = af + np.abs(r["face_attr"]); al = al + r["alpha"]
    return ae / len(idx), af / len(idx), al / len(idx)


def plot_global(ae, af, al, path):
    fig, ax = plt.subplots(1, 3, figsize=(17, 4), gridspec_kw={"width_ratios": [3, 2, 1]})
    im = ax[0].imshow(ae.T, cmap="viridis", aspect="auto")
    ax[0].set_xticks(range(N_CH)); ax[0].set_xticklabels(CH_NAMES, rotation=90, fontsize=7)
    ax[0].set_yticks(range(N_BANDS)); ax[0].set_yticklabels(BAND_NAMES)
    ax[0].set_title("Mean |attribution|: EEG"); plt.colorbar(im, ax=ax[0])
    top = np.argsort(af)[-8:]
    ax[1].barh([ff.FACE_FEATURE_NAMES[i] for i in top], af[top]); ax[1].set_title("Mean |attribution|: face (top 8)")
    ax[2].bar(["EEG", "Face"], al, color=["tab:purple", "tab:orange"]); ax[2].set_title("Avg modality weight")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


if __name__ == "__main__":
    from dataset import load_dataset
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="synthetic", choices=["synthetic", "deap"])
    ap.add_argument("--index", type=int, default=0)
    a = ap.parse_args()
    D, model = load_dataset(a.source), load_model(RUN_DIR / "model.pt")
    e, f = torch.tensor(D["eeg"][a.index]), torch.tensor(D["face"][a.index])
    res = explain(model, e, f)
    print("true:", CLASSES[D["y"][a.index]]); print("\n".join(describe(res, f.numpy())))
    plot_local(res, f.numpy(), RUN_DIR / "explain_local.png")
    plot_global(*global_importance(model, D["eeg"], D["face"]), RUN_DIR / "explain_global.png")
    print("saved figures in", RUN_DIR)
