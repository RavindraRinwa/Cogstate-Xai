# Multimodal Explainable AI for Real-Time Cognitive State Estimation (EEG + Face)

## 1. What it does
Estimates a person's **cognitive-affective state** every 1.5 s from two signals and explains every prediction.

| | |
|---|---|
| **Output state** | 4 quadrants of the arousal-valence plane: Bored/Sad, Calm/Content, Stressed/Angry, Excited/Happy (label = 2·[arousal>5] + [valence>5]) |
| **Modality 1: EEG** | 32 ch, 3 s windows (50 % overlap) -> Differential Entropy in theta/alpha/beta/gamma -> 32x4 map |
| **Modality 2: Face** | MediaPipe FaceMesh -> 10 interpretable per-frame cues (eye/mouth aspect ratio, brow raise/furrow, smile, head yaw/pitch) -> 21 window features (mean, std, blink count) |
| **Fusion** | Per-modality encoders -> attention-gated fusion -> classifier. Modality dropout in training, so it still works if the camera or EEG drops out |
| **XAI** | (a) learned modality weights, (b) Integrated Gradients (Captum) on EEG channel x band and on each face cue, (c) plain-text explanation, (d) global importance maps |
| **Real-time** | Webcam + EEG stream (simulated or LSL headset), 30 s personal calibration, live overlay with prediction + explanation |

```
EEG (32ch,128Hz) -> 3s window -> band-pass x4 -> DE (32x4) -> EEG encoder --\
                                                                              attention fusion -> state (4 classes)
Webcam -> FaceMesh -> 10 cues/frame -> 3s window -> 21 feats -> Face encoder -/        |
                                                              Integrated Gradients + modality weights -> explanation
```

## 2. Dataset
**Primary: DEAP** (Koelstra et al., 2012), the standard benchmark that has *both* EEG and face video.
- 32 participants x 40 one-minute music videos. 32-ch EEG (preprocessed version: 128 Hz, 4-45 Hz, EOG removed).
- Self-ratings of valence, arousal, dominance, liking (1-9) -> our labels.
- Frontal face video exists for **participants s01-s22 only**.
- Access: request via the EULA at https://www.eecs.qmul.ac.uk/mmv/datasets/deap/ . You need `data_preprocessed_python.zip` and the face video archive.

Place files as:
```
data/DEAP/data_preprocessed_python/s01.dat ... s32.dat
data/DEAP/face_video/**/s01_trial01.avi ...
```
**Alternatives:** MAHNOB-HCI (EEG + face video, 27 participants, EULA), SEED/DREAMER (EEG only), FER2013/AffectNet (face-only pretraining).
**No data yet?** `--source synthetic` generates data with a planted signal so you can run and verify the whole pipeline today.

> **Honest note on "cognitive state":** DEAP labels are emotional (arousal/valence). That is a legitimate, widely used proxy of cognitive-affective state (stress, engagement, boredom). For *mental workload / attention*, swap the label source for a workload dataset (e.g., STEW) and change `CLASSES` and `_label` in `dataset.py`; nothing else changes.

## 3. Run
```bash
pip install -r requirements.txt

# A) Smoke test, no downloads (≈1 min)
python train.py --source synthetic
python explain.py --source synthetic --index 3

# B) Real DEAP
python train.py --source deap            # first run extracts face features from videos (slow, ~hours on CPU; cached)
python train.py --source deap --no-face  # EEG-only on all 32 subjects (fast, no videos needed)
python explain.py --source deap --index 100

# C) Real-time demo
python realtime.py --eeg sim             # webcam + replayed EEG (pipeline demo)
python realtime.py --eeg lsl             # webcam + real headset via Lab Streaming Layer
python realtime.py --eeg none            # face only
```
Outputs go to `runs/`: `results.json` (CV ablation), `model.pt`, `explain_local.png`, `explain_global.png`.

## 4. Evaluation protocol
- **Subject-independent**: GroupKFold by participant (no subject appears in train and test) -> realistic for a deployed system.
- **Ablation**: EEG-only vs Face-only vs Fusion; also reports arousal and valence accuracy separately.
- Per-subject z-scoring is used (real-time replaces it with the 30 s calibration).
- Verified here on synthetic data: fusion > EEG-only > face-only, and IG recovers the planted EEG channels/bands and face cues. **These synthetic numbers are NOT results.**
- On real DEAP, expect modest numbers. Subject-independent 4-class is hard; roughly 35-50 % accuracy is typical, higher for binary arousal/valence. Report your own measured numbers.

## 5. Files
`config.py` settings | `eeg_features.py` DE features | `face_features.py` landmarks -> cues, video alignment | `dataset.py` DEAP/synthetic builder | `models.py` fusion network | `train.py` CV + ablation + final model | `explain.py` Integrated Gradients, plots, text | `realtime.py` live demo

## 6. Things to verify / limitations
1. **Video-EEG alignment**: `face_features.trial_face_windows` guesses whether a video still contains the 3 s baseline (video length ~63 s). Check on your copy of DEAP.
2. **Real-time EEG**: DEAP used a 32-ch research cap. Consumer headsets (4-14 ch) need retraining on that montage (edit `N_CH`/`CH_NAMES`). In `--eeg sim` mode the EEG is *not* from the person on camera.
3. **Calibration** in `realtime.py` is a heuristic replacement for per-subject normalization; use a calm-then-varied calibration for best results.
4. Attribution = what the model used, not proof of neuroscience causality. Face cues can be voluntarily masked, and self-report labels are noisy.
5. Ethics: emotion/cognitive-state inference is sensitive. Use only with informed consent, no high-stakes decisions.

## 7. Ideas to extend
CNN/Transformer on raw EEG windows, EEGNet branch, SHAP/GradCAM comparison, faithfulness metrics (deletion/insertion curves), cross-dataset test (train DEAP -> test MAHNOB-HCI), temporal smoothing with GRU.
