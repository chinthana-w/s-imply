from pathlib import Path
import json, cv2, typer
from rich import print
from .camera_discovery import discover_cameras, print_discovery
from .capture import CameraSource, VideoFileSource
from .config import load_config
from .detector import TapDetector
from .overlay import draw_overlay
from .recorder import JsonlRecorder
from .tracker_mediapipe import MediaPipeHandTracker
from .types import to_dict

app = typer.Typer()

@app.command("inspect-config")
def inspect_config(config:str="configs/default.yaml"):
    print(load_config(config))

@app.command("cameras")
def cameras(max_index:int=10, backend:str="auto"):
    print_discovery(discover_cameras(max_index, backend))
    print("On macOS, Android USB webcam may appear as a normal camera device when OS-exposed.")

def _run_loop(source, cfg, model_path, overlay_enabled=True, record=False, output="data/recordings/live.jsonl"):
    tracker=MediaPipeHandTracker(model_path=model_path)
    detector=TapDetector(cfg["detector"])
    rec = JsonlRecorder(output) if record else None
    while True:
        frame, ts=source.read()
        if frame is None: break
        hf = tracker.detect(frame, ts)
        result = detector.update(hf)
        for ev in result.events:
            print(json.dumps(to_dict(ev)))
        if rec:
            rec.write({"timestamp_ms":ts,"landmarks":[to_dict(l) for l in hf.landmarks],"world_landmarks":None if hf.world_landmarks is None else [to_dict(l) for l in hf.world_landmarks],"handedness":hf.handedness,"tracking_confidence":hf.tracking_confidence,"best_candidate":None if result.debug.best_candidate is None else to_dict(result.debug.best_candidate),"detector_state":result.debug.state,"normalized_distance":result.debug.normalized_distance,"event":to_dict(result.events[0]) if result.events else None})
        if overlay_enabled:
            cv2.imshow("digit9", draw_overlay(frame, result))
            k=cv2.waitKey(1)&0xFF
            if k in (27, ord('q')): break
    source.release();
    if rec: rec.close(); cv2.destroyAllWindows()

@app.command("live")
def live(config:str="configs/default.yaml",camera_index:int|None=None,backend:str|None=None,width:int|None=None,height:int|None=None,fps:int|None=None,model_path:str|None=None,no_overlay:bool=False,record:bool=False,output:str="data/recordings/live.jsonl",mirror:bool=False,rotate_degrees:int=0):
    cfg=load_config(config)
    cam=cfg["camera"]
    if camera_index is not None: cam["index"]=camera_index
    if backend is not None: cam["backend"]=backend
    if width is not None: cam["width"]=width
    if height is not None: cam["height"]=height
    if fps is not None: cam["target_fps"]=fps
    cam["mirror"] = mirror
    cam["rotate_degrees"] = rotate_degrees
    mp = model_path or cfg["tracker"]["model_path"]
    src=CameraSource(cam["index"],cam["width"],cam["height"],cam["target_fps"],cam["backend"],cam["mirror"],cam["rotate_degrees"]); src.start()
    _run_loop(src, cfg, mp, not no_overlay, record, output)

@app.command("video")
def video(input:str, config:str="configs/default.yaml", model_path:str|None=None, record:bool=False, output:str="data/recordings/video.jsonl", no_overlay:bool=False):
    cfg=load_config(config); mp=model_path or cfg["tracker"]["model_path"]; _run_loop(VideoFileSource(input), cfg, mp, not no_overlay, record, output)

@app.command("record")
def record_cmd(camera_index:int=0,label:str="",output:str="data/recordings/record.jsonl",duration_sec:float|None=None):
    _=label; _=duration_sec
    cfg=load_config("configs/default.yaml"); cfg["camera"]["index"]=camera_index
    src=CameraSource(index=camera_index); src.start(); _run_loop(src, cfg, cfg["tracker"]["model_path"], True, True, output)

if __name__ == "__main__":
    app()
