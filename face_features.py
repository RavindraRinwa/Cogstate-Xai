"""Interpretable facial-behaviour features from MediaPipe FaceMesh landmarks.
Per frame (10): eye-aspect-ratio L/R, mouth-aspect-ratio, brow raise L/R, brow furrow,
smile (lip-corner lift), mouth width, head yaw, head pitch.
Per 3 s window (21): mean + std of each + blink count."""
import hashlib
from pathlib import Path
import numpy as np
from config import *

BASE = ["ear_l", "ear_r", "mar", "brow_raise_l", "brow_raise_r", "brow_furrow",
        "smile", "mouth_width", "yaw", "pitch"]
FACE_FEATURE_NAMES = [f"{n}_mean" for n in BASE] + [f"{n}_std" for n in BASE] + ["blink_count"]
N_FACE = len(FACE_FEATURE_NAMES)

L_EYE, R_EYE = [33, 160, 158, 133, 153, 144], [362, 385, 387, 263, 373, 380]


def make_face_mesh():
    import mediapipe as mp
    return mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1,
                                           refine_landmarks=False, min_detection_confidence=0.5,
                                           min_tracking_confidence=0.5)


def landmarks_to_features(lm):
    """lm: (468, 2) pixel coordinates -> (10,)"""
    d = lambda a, b: float(np.linalg.norm(lm[a] - lm[b]))
    fh, fw = d(10, 152) + 1e-6, d(234, 454) + 1e-6
    ear = lambda i: (d(i[1], i[5]) + d(i[2], i[4])) / (2 * d(i[0], i[3]) + 1e-6)
    mouth_mid_y = (lm[13][1] + lm[14][1]) / 2
    corners_y = (lm[61][1] + lm[291][1]) / 2
    return np.array([
        ear(L_EYE), ear(R_EYE),
        d(13, 14) / (d(78, 308) + 1e-6),
        d(105, 159) / fh, d(334, 386) / fh,
        d(107, 336) / fw,
        (mouth_mid_y - corners_y) / fh,
        d(61, 291) / fw,
        (lm[1][0] - (lm[234][0] + lm[454][0]) / 2) / fw,
        (lm[1][1] - (lm[10][1] + lm[152][1]) / 2) / fh,
    ], dtype=np.float32)


def frame_features(mesh, frame_bgr):
    import cv2
    h, w = frame_bgr.shape[:2]
    res = mesh.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    if not res.multi_face_landmarks:
        return np.full(len(BASE), np.nan, np.float32)
    lm = np.array([[p.x * w, p.y * h] for p in res.multi_face_landmarks[0].landmark])
    return landmarks_to_features(lm)


def aggregate_window(F, min_valid=0.3):
    """F: (n_frames, 10) possibly with NaN rows -> (21,) or None if face mostly missing."""
    F = np.asarray(F, dtype=np.float64)
    valid = ~np.isnan(F).any(1)
    if len(F) < 5 or valid.mean() < min_valid:
        return None
    idx = np.arange(len(F))
    G = np.stack([np.interp(idx, idx[valid], F[valid, j]) for j in range(F.shape[1])], 1)
    closed = G[:, :2].mean(1) < BLINK_THR
    blinks = int((~closed[:-1] & closed[1:]).sum())
    return np.concatenate([G.mean(0), G.std(0), [blinks]]).astype(np.float32)


# ---------------- offline: DEAP face videos ----------------
def find_video(subject, trial):
    hits = list(FACE_DIR.rglob(f"s{subject:02d}_trial{trial:02d}.*"))   # .avi / .mp4 / ...
    return hits[0] if hits else None


def video_frame_features(path):
    import cv2
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 50.0
    step = max(1, int(round(fps / FACE_FPS)))
    mesh, ts, fs, i = make_face_mesh(), [], [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % step == 0:
            ts.append(i / fps)
            fs.append(frame_features(mesh, frame))
        i += 1
    cap.release()
    mesh.close()
    return np.array(ts), np.array(fs)


def cache_path(path):
    h = hashlib.md5(str(Path(path).resolve()).encode()).hexdigest()[:10]
    return CACHE_DIR / "face" / f"{Path(path).stem}_{h}.npz"


def trial_face_windows(path, n_win, offset=None):
    """Face features aligned to the EEG windows of one trial -> (n_win, N_FACE), NaN if missing.
    offset = seconds of video to skip (None -> DEAP heuristic)."""
    cache = cache_path(path)
    cache.parent.mkdir(exist_ok=True)
    if cache.exists():
        z = np.load(cache); t, F = z["t"], z["F"]
    else:
        t, F = video_frame_features(path)
        np.savez_compressed(cache, t=t, F=F)
    # Heuristic: if the video is ~63 s it still contains the 3 s baseline -> skip it. VERIFY on your copy.
    if offset is None:
        offset = BASELINE_SEC if len(t) and t[-1] > 61.5 else 0.0
    out = np.full((n_win, N_FACE), np.nan, np.float32)
    for i in range(n_win):
        a = offset + i * STEP_SEC
        m = (t >= a) & (t < a + WIN_SEC)
        g = aggregate_window(F[m]) if m.sum() >= 5 else None
        if g is not None:
            out[i] = g
    return out