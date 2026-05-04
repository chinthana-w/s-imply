from digit9.detector import TapDetector
from digit9.types import HandFrame, Landmark

def mk(dist,t):
    l=[Landmark(0,0,0) for _ in range(21)]; l[4]=Landmark(1,dist,0); l[5]=Landmark(1,0.1,0); l[6]=Landmark(1,0.2,0); l[9]=Landmark(2,0,0); l[17]=Landmark(3,0,0)
    return HandFrame(t,l,None,"left",1.0,640,480)

def test_sequence_tap():
    d=TapDetector({"contact_threshold":0.1,"release_threshold":0.15,"min_contact_frames":1,"max_tap_duration_ms":250,"debounce_ms":120,"history_frames":8,"smoothing_alpha":1.0,"ambiguous_margin":0.0})
    d.update(mk(0.3,0)); d.update(mk(0.01,10)); r=d.update(mk(0.3,20)); assert len(r.events)<=1
