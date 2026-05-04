# digit9
Python prototype for low-latency thumb-to-finger tap detection using MediaPipe Hand Landmarker landmarks plus a custom temporal detector.

## Why this repo
- Purpose: research/prototype for first-person, palm-facing thumb taps.
- Prototype vs final app: this repo validates algorithms quickly; final Meta-glasses-style app should be mobile-native for latency/power/integration.
- Python now: rapid iteration, easier tooling.
- Mobile later: Kotlin/Swift/NDK portability requires core logic decoupled from CV/UI libs.
- Hand Landmarker (not Gesture Recognizer): this project needs custom segment-level temporal logic.

## Install
```bash
pip install -e ".[dev]"
```

## Model setup
```bash
python scripts/download_mediapipe_model.py
```
Place model at `models/hand_landmarker.task`.

## Android USB on macOS
Android 14 QPR1+ compatible devices may expose USB webcam/UVC. Camera index can vary.
Run camera discovery first.

## CLI
```bash
digit9 inspect-config
digit9 cameras
digit9 live --camera-index 1 --backend avfoundation
digit9 video --input sample.mp4
digit9 live --record --output data/recordings/live.jsonl
pytest
```

## Architecture overview
- Portable core: `types.py`, `geometry.py`, `state_machine.py`, `detector.py`
- Integrations: capture/tracker/overlay/recorder/app

## Known limitations
- thumb occlusion at contact
- MediaPipe landmark jitter
- motion blur
- camera index instability on macOS
- Android phone may not expose as UVC on all devices/OS versions
- segment classification may need calibration or learned temporal classifier later
