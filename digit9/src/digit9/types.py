from dataclasses import asdict, dataclass, field


@dataclass
class Landmark:
    x: float
    y: float
    z: float


@dataclass
class HandFrame:
    timestamp_ms: int
    landmarks: list[Landmark]
    world_landmarks: list[Landmark] | None
    handedness: str
    tracking_confidence: float
    frame_width: int
    frame_height: int


@dataclass
class CandidateContact:
    finger: str
    segment: str
    distance: float
    normalized_distance: float
    closest_point: Landmark
    confidence: float
    ambiguous: bool = False


@dataclass
class TapEvent:
    event_type: str
    finger: str
    segment: str
    timestamp_ms: int
    confidence: float
    latency_ms: float
    handedness: str


@dataclass
class DetectorDebugInfo:
    state: str = "IDLE"
    normalized_distance: float | None = None
    best_candidate: CandidateContact | None = None
    note: str = ""


@dataclass
class DetectorResult:
    events: list[TapEvent] = field(default_factory=list)
    debug: DetectorDebugInfo = field(default_factory=DetectorDebugInfo)


def to_dict(value: object) -> dict:
    return asdict(value)
