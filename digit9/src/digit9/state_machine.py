from dataclasses import dataclass
from .types import CandidateContact, TapEvent

@dataclass
class TapStateMachine:
    config: dict
    state: str = "IDLE"
    contact_start_ms: int | None = None
    contact_frames: int = 0
    cooldown_until_ms: int = 0
    prev_distance: float | None = None

    def reset(self, reason: str = ""):
        self.state, self.contact_start_ms, self.contact_frames, self.prev_distance = "IDLE", None, 0, None

    def get_debug_info(self):
        return {"state": self.state}

    def update(self, candidate: CandidateContact | None, timestamp_ms: int, handedness: str, tracking_confidence: float):
        if tracking_confidence < 0.2 or candidate is None:
            self.reset("tracking_lost")
            return None
        if timestamp_ms < self.cooldown_until_ms:
            self.state = "COOLDOWN"
            return None
        if candidate.ambiguous:
            self.reset("ambiguous")
            return None
        d = candidate.normalized_distance
        if self.state == "IDLE":
            if self.prev_distance is not None and d < self.prev_distance:
                self.state = "APPROACHING"
        elif self.state == "APPROACHING":
            if d < self.config["contact_threshold"]:
                self.contact_frames += 1
                if self.contact_frames >= self.config["min_contact_frames"]:
                    self.state = "CONTACT"
                    self.contact_start_ms = timestamp_ms
            else:
                self.contact_frames = 0
        elif self.state == "CONTACT" and d > self.config["release_threshold"]:
            self.state = "RELEASED"
            dur = timestamp_ms - (self.contact_start_ms or timestamp_ms)
            if dur <= self.config["max_tap_duration_ms"]:
                ev = TapEvent("tap", candidate.finger, candidate.segment, timestamp_ms, candidate.confidence, 0.0, handedness)
                self.cooldown_until_ms = timestamp_ms + self.config["debounce_ms"]
                self.state = "COOLDOWN"
                self.prev_distance = d
                return ev
            self.reset("too_long")
        self.prev_distance = d
        return None
