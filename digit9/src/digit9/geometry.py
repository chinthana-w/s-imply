import math
import numpy as np
from .landmarks import FINGERS, SEGMENTS, THUMB_TIP
from .types import CandidateContact, HandFrame, Landmark


def _arr(l: Landmark) -> np.ndarray:
    return np.array([l.x, l.y, l.z], dtype=float)


def closest_point_on_segment_3d(point, a, b):
    p, a, b = np.array(point, float), np.array(a, float), np.array(b, float)
    ab = b - a
    d = np.dot(ab, ab)
    if not np.isfinite(d) or d <= 1e-12:
        return a
    t = np.dot(p - a, ab) / d
    return a + np.clip(t, 0.0, 1.0) * ab


def point_to_segment_distance_3d(point, a, b):
    c = closest_point_on_segment_3d(point, a, b)
    return float(np.linalg.norm(np.array(point) - c)), c


def compute_palm_width(landmarks):
    return float(np.linalg.norm(_arr(landmarks[5]) - _arr(landmarks[17]))) if len(landmarks) > 17 else 0.0


def compute_wrist_to_middle_mcp(landmarks):
    return float(np.linalg.norm(_arr(landmarks[0]) - _arr(landmarks[9]))) if len(landmarks) > 9 else 0.0


def compute_hand_scale(landmarks):
    return max(compute_palm_width(landmarks), compute_wrist_to_middle_mcp(landmarks), 1e-6)


def normalize_landmarks(landmarks):
    s = compute_hand_scale(landmarks)
    return [Landmark(l.x / s, l.y / s, l.z / s) for l in landmarks] if s > 0 else landmarks


def compute_thumb_tip_to_segments(hand_frame: HandFrame):
    points = hand_frame.world_landmarks or hand_frame.landmarks
    if len(points) < 21:
        return []
    thumb = _arr(points[THUMB_TIP])
    out = []
    for finger, idxs in FINGERS.items():
        for segment, (i0, i1) in SEGMENTS.items():
            a, b = _arr(points[idxs[i0]]), _arr(points[idxs[i1]])
            dist, cp = point_to_segment_distance_3d(thumb, a, b)
            out.append((finger, segment, dist, cp))
    return out


def select_best_contact_candidate(hand_frame: HandFrame, detector_config):
    cands = compute_thumb_tip_to_segments(hand_frame)
    if not cands:
        return None, None
    scale = compute_hand_scale(hand_frame.world_landmarks or hand_frame.landmarks)
    if not math.isfinite(scale) or scale <= 1e-9:
        return None, None
    scored = sorted(cands, key=lambda x: x[2])
    best, second = scored[0], scored[1] if len(scored) > 1 else scored[0]
    nbest, nsecond = best[2] / scale, second[2] / scale
    ambiguous = (nsecond - nbest) < float(detector_config.get("ambiguous_margin", 0.018))
    conf = max(0.0, min(1.0, 1.0 - nbest * 4))
    b = CandidateContact(best[0], best[1], float(best[2]), float(nbest), Landmark(*best[3]), conf, ambiguous)
    s = CandidateContact(second[0], second[1], float(second[2]), float(nsecond), Landmark(*second[3]), conf, ambiguous)
    return b, s
