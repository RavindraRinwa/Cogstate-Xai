"""Subject-independent evaluation (GroupKFold over participants) + ablation (EEG / Face / Fusion),
then trains the final fusion model on all subjects and saves runs/model.pt"""
import argparse, json
import numpy as np, torch, torch.nn as nn
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score
from config import *
from dataset import load_dataset
from models import MultimodalNet, fixed_mask, random_mask
import face_features as ff


def fit(model, E, F, y, mods, epochs, dev, bs=128, lr=1e-3):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    lossf, n = nn.CrossEntropyLoss(label_smoothing=0.1), len(y)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n - bs + 1, bs):
            idx = perm[i:i + bs]
            e, f, t = E[idx].to(dev), F[idx].to(dev), y[idx].to(dev)
            m = random_mask(len(idx), .25, dev) if mods == "both" else fixed_mask(mods, len(idx), dev)
            logits, _ = model(e + .05 * torch.randn_like(e), f + .05 * torch.randn_like(f), m)
            opt.zero_grad(); lossf(logits, t).backward(); opt.step()
        sched.step()
    return model


@torch.no_grad()
def predict(model, E, F, mods, dev):
    model.eval()
    logits, a = model(E.to(dev), F.to(dev), fixed_mask(mods, len(E), dev))
    return logits.argmax(1).cpu().numpy(), a.cpu().numpy()


def metrics(y, p):
    return dict(acc=(y == p).mean(), f1=f1_score(y, p, average="macro"),
                arousal_acc=((y >> 1) == (p >> 1)).mean(), valence_acc=((y & 1) == (p & 1)).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="synthetic", choices=["synthetic", "deap"])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--no-face", action="store_true", help="EEG-only dataset (DEAP s01-s32, no videos)")
    ap.add_argument("--no-final", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    D = load_dataset(a.source, use_face=True)
    E, F, y = torch.tensor(D["eeg"]), torch.tensor(D["face"]), torch.tensor(D["y"])
    subs = np.unique(D["subj"])
    print(f"{len(y)} windows | {len(subs)} subjects | class counts {np.bincount(D['y'], minlength=4)}")

    results = {}
    for mods in (["eeg"] if a.no_face else ["eeg", "face", "both"]):
        rows = []
        for k, (tr, te) in enumerate(GroupKFold(min(a.folds, len(subs))).split(E, y, D["subj"])):
            m = fit(MultimodalNet(ff.N_FACE).to(dev), E[tr], F[tr], y[tr], mods, a.epochs, dev)
            p, _ = predict(m, E[te], F[te], mods, dev)
            rows.append(metrics(D["y"][te], p))
        results[mods] = {k: (float(np.mean([r[k] for r in rows])), float(np.std([r[k] for r in rows])))
                         for k in rows[0]}
        print(f"{mods:5s} " + "  ".join(f"{k} {v[0]:.3f}±{v[1]:.3f}" for k, v in results[mods].items()))
    json.dump(results, open(RUN_DIR / "results.json", "w"), indent=2)

    if not a.no_final:
        mods = "eeg" if a.no_face else "both"
        m = fit(MultimodalNet(ff.N_FACE).to(dev), E, F, y, mods, a.epochs, dev).cpu()
        torch.save({"state": m.state_dict(), "n_face": ff.N_FACE}, RUN_DIR / "model.pt")
        print("saved", RUN_DIR / "model.pt")


if __name__ == "__main__":
    main()
