from digit9.geometry import closest_point_on_segment_3d, point_to_segment_distance_3d, compute_hand_scale, select_best_contact_candidate
from digit9.types import HandFrame, Landmark

def test_closest_inside_segment():
    c=closest_point_on_segment_3d([0.5,0,0],[0,0,0],[1,0,0]); assert abs(c[0]-0.5)<1e-6

def test_clamped_endpoint():
    c=closest_point_on_segment_3d([2,0,0],[0,0,0],[1,0,0]); assert abs(c[0]-1)<1e-6

def test_zero_length_segment():
    d,_=point_to_segment_distance_3d([1,0,0],[0,0,0],[0,0,0]); assert d>0

def test_hand_scale():
    l=[Landmark(0,0,0) for _ in range(21)]; l[5]=Landmark(1,0,0); l[17]=Landmark(3,0,0); assert compute_hand_scale(l)>0

def test_synthetic_candidate_and_ambiguous():
    l=[Landmark(0,0,0) for _ in range(21)]; l[4]=Landmark(1.0,0.0,0.0); l[5]=Landmark(1.0,0.1,0.0); l[6]=Landmark(1.0,0.2,0.0); l[17]=Landmark(3,0,0); l[9]=Landmark(2,0,0)
    hf=HandFrame(0,l,None,"left",1.0,640,480)
    best,_=select_best_contact_candidate(hf,{"ambiguous_margin":0.5}); assert best is not None
