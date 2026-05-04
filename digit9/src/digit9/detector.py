from collections import deque
from .geometry import select_best_contact_candidate
from .state_machine import TapStateMachine
from .types import DetectorDebugInfo, DetectorResult, HandFrame

class TapDetector:
    def __init__(self, config: dict):
        self.config = config
        self.state_machine = TapStateMachine(config)
        self.history = deque(maxlen=config.get("history_frames", 8))
        self.ema = None

    def update(self, frame: HandFrame) -> DetectorResult:
        best, second = select_best_contact_candidate(frame, self.config)
        if best is not None:
            a = self.config.get("smoothing_alpha", 0.45)
            self.ema = best.normalized_distance if self.ema is None else a * best.normalized_distance + (1 - a) * self.ema
            best.normalized_distance = self.ema
        ev = self.state_machine.update(best, frame.timestamp_ms, frame.handedness, frame.tracking_confidence)
        dbg = DetectorDebugInfo(state=self.state_machine.state, normalized_distance=(best.normalized_distance if best else None), best_candidate=best)
        return DetectorResult(events=([ev] if ev else []), debug=dbg)
