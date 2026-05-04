import cv2

def draw_overlay(frame, result, fps=0.0, recording=False):
    cv2.putText(frame, f"state={result.debug.state} dist={result.debug.normalized_distance}", (20,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    cv2.putText(frame, f"fps={fps:.1f} rec={recording}", (20,55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
    return frame
