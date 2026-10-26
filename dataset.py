"""Builds the aligned multimodal dataset: EEG DE features + facial features + 4-class label.
source="deap"      -> real DEAP EEG (+ face videos s01-s22)
source="synthetic" -> planted-signal fake data so the whole pipeline runs without downloads."""
import pickle
import numpy as np
from config import *
from eeg_features import trial_de_windows
import face_features as ff


def _label(valence, arousal):
    return 2 * int(arousal > 5) + int(valence > 5)


def zscore_by_subject(x, subj):
    x = x.copy()
    for s in np.unique(subj):
        m = subj == s
        x[m] = (x[m] - x[m].mean(0)) / (x[m].std(0) + 1e-6)
    return x


def load_deap_subject(s, use_face=True):
    f = CACHE_DIR / f"deap_s{s:02d}_{'mm' if use_face else 'eeg'}.npz"
    if f.exists():
        z = np.load(f); return {k: z[k] for k in z.files}
    p = DEAP_DIR / f"s{s:02d}.dat"
    if not p.exists():
        return None
    with open(p, "rb") as fh:
        d = pickle.load(fh, encoding="latin1")
    E, F, Y, T = [], [], [], []
    for t in range(40):
        de = trial_de_windows(d["data"][t, :N_CH, BASELINE_SEC * FS:])
        y = _label(*d["labels"][t, :2])           # valence, arousal
        if use_face:
            vp = ff.find_video(s, t + 1)
            if vp is None:
                continue
            fw = ff.trial_face_windows(vp, len(de))
            ok = ~np.isnan(fw).any(1)
            de, fw = de[ok], fw[ok]
            F.append(fw)
        E.append(de); Y.append(np.full(len(de), y)); T.append(np.full(len(de), t))
    if not E:
        return None
    y = np.concatenate(Y)
    out = dict(eeg=np.concatenate(E), y=y, trial=np.concatenate(T),
               face=np.concatenate(F) if use_face else np.zeros((len(y), ff.N_FACE), np.float32))
    np.savez_compressed(f, **out)
    return out


def make_synthetic(n_subj=10, n_trials=40, n_win=39, seed=0):
    """Planted ground truth (so XAI can be sanity-checked):
       valence  -> frontal alpha asymmetry (F4 vs F3) + smile
       arousal  -> frontal beta/gamma up, occipital alpha down + brow raise, mouth opening, fewer blinks"""
    rng = np.random.default_rng(seed)
    nm = ff.FACE_FEATURE_NAMES.index
    parts = {}
    for s in range(1, n_subj + 1):
        so = rng.normal(0, 1, (1, N_CH, N_BANDS)); fo = rng.normal(0, 1, (1, ff.N_FACE))
        E, F, Y, T = [], [], [], []
        for t in range(n_trials):
            y = int(rng.integers(0, 4)); hv, ha = (y & 1) * 2 - 1, (y >> 1) * 2 - 1
            e = rng.normal(0, 1, (n_win, N_CH, N_BANDS)) + so + rng.normal(0, .6, (1, N_CH, N_BANDS))
            e[:, 19, 1] += .7 * hv; e[:, 2, 1] -= .7 * hv
            e[:, [0, 16, 2, 19], 2:] += .7 * ha; e[:, [13, 14, 31], 1] -= .7 * ha
            f = rng.normal(0, 1, (n_win, ff.N_FACE)) + fo + rng.normal(0, .6, (1, ff.N_FACE))
            f[:, nm("smile_mean")] += .8 * hv
            for n in ("brow_raise_l_mean", "brow_raise_r_mean", "mar_mean"):
                f[:, nm(n)] += .6 * ha
            f[:, nm("blink_count")] -= .5 * ha
            E.append(e); F.append(f); Y.append(np.full(n_win, y)); T.append(np.full(n_win, t))
        parts[s] = dict(eeg=np.concatenate(E).astype(np.float32), face=np.concatenate(F).astype(np.float32),
                        y=np.concatenate(Y), trial=np.concatenate(T))
    return parts


def load_dataset(source="synthetic", subjects=None, use_face=True):
    if source == "synthetic":
        parts = make_synthetic()
    else:
        subjects = subjects or list(range(1, 23 if use_face else 33))
        parts = {s: load_deap_subject(s, use_face) for s in subjects}
        parts = {s: p for s, p in parts.items() if p is not None}
        if not parts:
            raise SystemExit(f"No DEAP data found under {DEAP_DIR} (see README for download steps).")
    subj = np.concatenate([np.full(len(p["y"]), s) for s, p in parts.items()])
    eeg = zscore_by_subject(np.concatenate([p["eeg"] for p in parts.values()]), subj)
    face = zscore_by_subject(np.concatenate([p["face"] for p in parts.values()]), subj)
    return dict(eeg=eeg.astype(np.float32), face=face.astype(np.float32),
                y=np.concatenate([p["y"] for p in parts.values()]).astype(np.int64),
                subj=subj, use_face=use_face)
