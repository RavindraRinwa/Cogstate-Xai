"""Real-time demo: webcam (face) + EEG stream -> 4-state prediction + live explanation overlay.
EEG sources: sim (replays a DEAP trial/noise; DEMO ONLY - EEG is not from the person on camera),
             lsl (live headset through Lab Streaming Layer; must be 32 ch @ 128 Hz in DEAP channel order),
             none (face only).
The first --calib-sec seconds are used as a personal baseline (replaces per-subject z-scoring)."""
import argparse, collections, pickle, time
import numpy as np, cv2, torch
from scipy.signal import resample
from config import *
import face_features as ff
from eeg_features import window_de
from models import load_model
from explain import explain, describe
from mindwave import MindWaveEEG


class SimEEG:
    def __init__(self):
        p = DEAP_DIR / "s01.dat"
        if p.exists():
            with open(p, "rb") as fh:
                self.x = pickle.load(fh, encoding="latin1")["data"][0, CH_IDX, BASELINE_SEC * FS:].T
        else:
            self.x = np.random.randn(60 * FS, N_CH) * 10
        self.t0, self.n = time.time(), 0

    def read(self):
        tgt = int((time.time() - self.t0) * FS)
        out = self.x[np.arange(self.n, tgt) % len(self.x)]
        self.n = tgt
        return out


class LSLEEG:
    """Any headset/board that publishes an LSL stream of type EEG (OpenBCI GUI, Muse via muselsl, ...)."""
    def __init__(self, chan):
        from pylsl import StreamInlet, resolve_byprop
        s = resolve_byprop("type", "EEG", timeout=10)
        if not s:
            raise RuntimeError("No LSL EEG stream found")
        self.inl, self.chan = StreamInlet(s[0]), chan
        self.fs = s[0].nominal_srate()

    def read(self):
        c, _ = self.inl.pull_chunk(timeout=0.0)
        return np.asarray(c, dtype=np.float32)[:, self.chan] if c else np.zeros((0, N_CH))


class SerialEEG:
    """USB/serial boards (Arduino + BioAmp, etc.) that print one line per sample: 'v1,v2' (or 'v1 v2')."""
    def __init__(self, port, baud, chan, fs):
        import serial
        self.ser, self.chan, self.fs = serial.Serial(port, baud, timeout=0), chan, fs
        self.rest = b""

    def read(self):
        self.rest += self.ser.read(self.ser.in_waiting or 1)
        *lines, self.rest = self.rest.split(b"\n")
        out = []
        for ln in lines:
            try:
                v = [float(x) for x in ln.replace(b",", b" ").split()]
                out.append([v[i] for i in self.chan])
            except (ValueError, IndexError):
                pass                                          # skip partial/garbled lines
        return np.asarray(out, dtype=np.float32).reshape(-1, N_CH)


def stats(xs):
    mu, sd = np.mean(xs, 0), np.std(xs, 0)
    return mu, np.maximum(sd, 0.1 * np.abs(mu) + 1e-3)      # floor: a calm calibration has tiny variance


def draw(frame, ps, lines, status):
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 190), (0, 0, 0), -1)
    cv2.putText(frame, status, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 255, 255), 1)
    for i, (c, p) in enumerate(zip(CLASSES, ps)):
        cv2.putText(frame, c, (10, 45 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
        cv2.rectangle(frame, (150, 33 + 20 * i), (150 + int(200 * p), 47 + 20 * i), (0, 200, 0), -1)
    for i, l in enumerate(lines[1:]):
        cv2.putText(frame, l, (370 if frame.shape[1] > 900 else 10, 45 + 20 * i + (0 if frame.shape[1] > 900 else 90)),
                    cv2.FONT_HERSHEY_SIMPLEX, .45, (200, 200, 255), 1)
    cv2.putText(frame, lines[0], (10, 183), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 255, 0), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(RUN_DIR / "model.pt"))
    ap.add_argument("--eeg", default="sim", choices=["sim", "lsl", "serial", "mindwave", "none"])
    ap.add_argument("--chan", type=int, nargs="+", default=list(range(N_CH)),
                    help="device channel indices, in the order of USE_CHANNELS (config.py)")
    ap.add_argument("--fs", type=float, default=0, help="device sampling rate in Hz (auto for LSL)")
    ap.add_argument("--port", default=""); ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--calib-sec", type=float, default=30)
    ap.add_argument("--cam", type=int, default=0)
    a = ap.parse_args()

    model = load_model(a.ckpt)
    if a.eeg == "mindwave" and N_CH != 1:
        raise SystemExit('MindWave has ONE EEG channel (Fp1): set USE_CHANNELS = ["Fp1"] in config.py, retrain, then rerun.')
    src = {"sim": SimEEG, "lsl": lambda: LSLEEG(a.chan),
           "serial": lambda: SerialEEG(a.port, a.baud, a.chan, a.fs or FS),
           "mindwave": lambda: MindWaveEEG(a.port, a.baud), "none": lambda: None}[a.eeg]()
    dev_fs = a.fs or getattr(src, "fs", FS)
    mesh, cap = ff.make_face_mesh(), cv2.VideoCapture(a.cam)
    eeg_buf, face_buf = collections.deque(maxlen=int(WIN_SEC * dev_fs)), collections.deque()
    cal_e, cal_f, st = [], [], None
    last_step, last_face, ps, lines = time.time(), 0.0, np.ones(len(CLASSES)) / len(CLASSES), ["warming up"] * 5

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        now = time.time()
        if src is not None:
            eeg_buf.extend(src.read())
        if now - last_face >= 1 / FACE_FPS:
            last_face = now
            face_buf.append((now, ff.frame_features(mesh, frame)))
        while face_buf and face_buf[0][0] < now - WIN_SEC:
            face_buf.popleft()

        if now - last_step >= STEP_SEC:
            last_step = now
            de = None
            if src is not None and len(eeg_buf) == eeg_buf.maxlen and getattr(src, "poor", 0) <= 50:
                x = np.asarray(eeg_buf, dtype=np.float64).T
                if int(dev_fs) != FS:
                    x = resample(x, WIN, axis=-1)                # device rate -> 128 Hz (the model's rate)
                de = window_de(x)
            fw = ff.aggregate_window(np.array([f for _, f in face_buf]))
            if st is None:                                            # ---- calibration phase
                if fw is not None and (de is not None or src is None):
                    cal_f.append(fw); cal_e.append(de if de is not None else np.zeros((N_CH, N_BANDS)))
                if len(cal_f) >= a.calib_sec / STEP_SEC:
                    st = (*stats(cal_e), *stats(cal_f))
            else:                                                     # ---- live inference + explanation
                mu_e, sd_e, mu_f, sd_f = st
                e = np.clip((de - mu_e) / sd_e, -4, 4) if de is not None else np.zeros((N_CH, N_BANDS))
                f = np.clip((fw - mu_f) / sd_f, -4, 4) if fw is not None else np.zeros(ff.N_FACE)
                mask = (de is not None, fw is not None)
                if any(mask):
                    r = explain(model, torch.tensor(e), torch.tensor(f), mask, n_steps=16)
                    ps = .6 * ps + .4 * r["probs"]
                    lines = describe(r, f)
        status = (f"CALIBRATING {len(cal_f)}/{int(a.calib_sec / STEP_SEC)} - stay neutral" if st is None
                  else "LIVE (EEG: %s%s)" % (a.eeg, " - POOR CONTACT, EEG ignored" if getattr(src, "poor", 0) > 50 else ""))
        draw(frame, ps, lines, status)
        cv2.imshow("Cognitive state + explanation (q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release(); cv2.destroyAllWindows()


if __name__ == "__main__":
    main()