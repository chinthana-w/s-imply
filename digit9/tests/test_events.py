from dataclasses import asdict
from digit9.types import TapEvent, CandidateContact, HandFrame, Landmark

def test_tap_event_serialization():
    ev=TapEvent("tap","index","proximal",1,0.9,5.0,"left"); assert asdict(ev)["finger"]=="index"

def test_candidate_serialization():
    c=CandidateContact("index","middle",1,0.1,Landmark(0,0,0),0.8); assert asdict(c)["segment"]=="middle"

def test_handframe_construction():
    h=HandFrame(1,[Landmark(0,0,0)],None,"unknown",0.0,1,1); assert h.frame_width==1
