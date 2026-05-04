from pathlib import Path
from .types import HandFrame
class MediaPipeHandTracker:
    def __init__(self, model_path:str, **kwargs):
        if not Path(model_path).exists():
            raise FileNotFoundError("Missing model. Run scripts/download_mediapipe_model.py and place at models/hand_landmarker.task")
    def detect(self, frame, timestamp_ms:int):
        return HandFrame(timestamp_ms, [], None, "unknown", 0.0, frame.shape[1], frame.shape[0])
