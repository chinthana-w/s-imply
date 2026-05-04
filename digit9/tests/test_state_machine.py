from digit9.state_machine import TapStateMachine
from digit9.types import CandidateContact, Landmark

def cand(d,amb=False): return CandidateContact("index","proximal",d,d,Landmark(0,0,0),0.9,amb)

def test_hover_no_emit():
    sm=TapStateMachine({"contact_threshold":0.1,"release_threshold":0.15,"min_contact_frames":2,"max_tap_duration_ms":250,"debounce_ms":120})
    assert sm.update(cand(0.2),0,"left",1.0) is None

def test_emit_one_tap_and_cooldown():
    sm=TapStateMachine({"contact_threshold":0.1,"release_threshold":0.15,"min_contact_frames":1,"max_tap_duration_ms":250,"debounce_ms":120})
    sm.update(cand(0.2),0,"left",1); sm.update(cand(0.05),10,"left",1); ev=sm.update(cand(0.2),30,"left",1); assert ev is not None
    assert sm.update(cand(0.05),40,"left",1) is None

def test_hand_loss_reset():
    sm=TapStateMachine({"contact_threshold":0.1,"release_threshold":0.15,"min_contact_frames":1,"max_tap_duration_ms":250,"debounce_ms":120}); sm.update(None,0,"left",0.0); assert sm.state=="IDLE"

def test_ambiguous_suppressed():
    sm=TapStateMachine({"contact_threshold":0.1,"release_threshold":0.15,"min_contact_frames":1,"max_tap_duration_ms":250,"debounce_ms":120}); assert sm.update(cand(0.05,True),0,"left",1) is None
