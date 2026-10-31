from pathlib import Path

USE_CHANNELS = ["Fp1"]
ROOT = Path(__file__).resolve().parent
DEAP_DIR = ROOT / "data" / "DEAP" / "deap-dataset" / "data_preprocessed_python"
# DEAP_DIR = ROOT / "data" / "DEAP" / "data_preprocessed_python"   # s01.dat ... s32.dat
FACE_DIR = ROOT / "data" / "DEAP" / "face_video"                 # s01_trial01.avi ... (s01-s22 only)
CACHE_DIR = ROOT / "cache"
RUN_DIR = ROOT / "runs"
for _d in (CACHE_DIR, RUN_DIR):
    _d.mkdir(exist_ok=True)

# ---- EEG (DEAP preprocessed: 128 Hz, 4-45 Hz band-pass, EOG removed) ----
FS = 128
BASELINE_SEC = 3                       # first 3 s of each trial are pre-trial baseline
WIN_SEC, STEP_SEC = 3.0, 1.5           # sliding window (50 % overlap)
WIN, STEP = int(WIN_SEC * FS), int(STEP_SEC * FS)
BANDS = {"theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 45)}
N_CH, N_BANDS = 32, len(BANDS)
CH_NAMES = ["Fp1", "AF3", "F3", "F7", "FC5", "FC1", "C3", "T7", "CP5", "CP1", "P3", "P7",
            "PO3", "O1", "Oz", "Pz", "Fp2", "AF4", "Fz", "F4", "F8", "FC6", "FC2", "Cz",
            "C4", "T8", "CP6", "CP2", "P4", "P8", "PO4", "O2"]

# ---- Cognitive-affective state = quadrant of the arousal-valence plane ----
# label = 2 * (arousal > 5) + (valence > 5)
CLASSES = ["Bored/Sad", "Calm/Content", "Stressed/Angry", "Excited/Happy"]

# ---- Face ----
FACE_FPS = 15          # frames/s analysed (both offline and real-time)
BLINK_THR = 0.20       # eye-aspect-ratio threshold for a blink
