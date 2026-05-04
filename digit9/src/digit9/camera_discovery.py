import cv2
from rich.console import Console
from rich.table import Table

def discover_cameras(max_index=10, backend="auto"):
    rows=[]
    api = cv2.CAP_AVFOUNDATION if backend=="avfoundation" and hasattr(cv2, "CAP_AVFOUNDATION") else cv2.CAP_ANY
    for i in range(max_index+1):
        cap=cv2.VideoCapture(i, api); opened=cap.isOpened(); ok=False; w=h=fps=0
        if opened:
            ok,_=cap.read(); w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); fps=float(cap.get(cv2.CAP_PROP_FPS)); cap.release()
        rows.append((i, opened, ok, w, h, fps))
    return rows

def print_discovery(rows):
    t=Table(title="Camera discovery")
    for c in ["Index","Opened","Frame","Width","Height","FPS"]: t.add_column(c)
    for r in rows: t.add_row(*map(str,r))
    Console().print(t)
