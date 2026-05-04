import time, cv2
class CameraSource:
    def __init__(self, index=0, width=1280, height=720, fps=60, backend="auto", mirror=False, rotate_degrees=0):
        self.index=index; self.width=width; self.height=height; self.fps=fps; self.backend=backend; self.mirror=mirror; self.rotate=rotate_degrees; self.cap=None
    def start(self):
        api = cv2.CAP_AVFOUNDATION if self.backend=="avfoundation" and hasattr(cv2,"CAP_AVFOUNDATION") else cv2.CAP_ANY
        self.cap=cv2.VideoCapture(self.index, api)
        if not self.cap.isOpened(): raise RuntimeError(f"Cannot open camera index {self.index}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width); self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height); self.cap.set(cv2.CAP_PROP_FPS, self.fps)
    def read(self):
        ok, frame = self.cap.read()
        if not ok: return None, int(time.monotonic()*1000)
        if self.mirror: frame=cv2.flip(frame,1)
        if self.rotate in (90,180,270): frame={90:cv2.ROTATE_90_CLOCKWISE,180:cv2.ROTATE_180,270:cv2.ROTATE_90_COUNTERCLOCKWISE}.get(self.rotate) and cv2.rotate(frame,{90:cv2.ROTATE_90_CLOCKWISE,180:cv2.ROTATE_180,270:cv2.ROTATE_90_COUNTERCLOCKWISE}[self.rotate])
        return frame, int(time.monotonic()*1000)
    def release(self):
        if self.cap: self.cap.release()

class VideoFileSource:
    def __init__(self, path:str): self.cap=cv2.VideoCapture(path)
    def read(self):
        ok, frame = self.cap.read()
        return (frame, int(time.monotonic()*1000)) if ok else (None, None)
    def release(self): self.cap.release()
