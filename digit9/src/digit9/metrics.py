import time
class FPSCounter:
    def __init__(self): self.prev=time.monotonic(); self.fps=0.0
    def tick(self):
        now=time.monotonic(); dt=now-self.prev; self.prev=now
        if dt>0: self.fps=0.9*self.fps+0.1*(1.0/dt)
        return self.fps
