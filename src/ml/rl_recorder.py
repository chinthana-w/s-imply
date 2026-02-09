
import torch
import pickle
import os
import uuid
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class ExperienceStep:
    # State data (inputs to model)
    node_ids: Any = None
    mask_valid: Any = None
    gate_types: Any = None
    files: Any = None
    
    # Action data
    # We store the *assignments* chosen by the model
    # logits? For REINFORCE/PPO we need logits + action mask? 
    # For now, let's store the raw inputs so we can re-run the forward pass 
    # and the logic assignment selected dict {gate_id: val}
    selected_assignment: Dict[int, int] = field(default_factory=dict)
    
    # Metadata
    pair_info: Dict[str, Any] = field(default_factory=dict)
    
    # Outcome
    # Reward will be populated later based on backtracks/success
    reward: float = 0.0
    
    # For debugging/tracking
    step_id: str = field(default_factory=lambda: str(uuid.uuid4()))

class ExperienceRecorder:
    def __init__(self, save_dir: str = "data/rl_experience", buffer_size: int = 10):
        self.save_dir = save_dir
        self.buffer_size = buffer_size
        self.current_episode_steps: List[ExperienceStep] = []
        self.episode_buffer: List[List[ExperienceStep]] = []
        self.episode_id = str(uuid.uuid4())
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
            
    def start_episode(self, episode_id: str = None):
        self.episode_id = episode_id or str(uuid.uuid4())
        self.current_episode_steps = []
        
    def log_step(self, 
                 node_ids, 
                 mask_valid, 
                 gate_types, 
                 files, 
                 pair_info,
                 selected_assignment):
        
        # Clone tensors to CPU to save memory on GPU/avoid referencing active graph
        step = ExperienceStep(
            node_ids=node_ids.cpu().detach() if isinstance(node_ids, torch.Tensor) else node_ids,
            mask_valid=mask_valid.cpu().detach() if isinstance(mask_valid, torch.Tensor) else mask_valid,
            gate_types=gate_types.cpu().detach() if isinstance(gate_types, torch.Tensor) else gate_types,
            files=files,
            pair_info=pair_info,
            selected_assignment=selected_assignment
        )
        self.current_episode_steps.append(step)
        return step
        
    def mark_backtrack(self, step_index: int = -1, penalty: float = -0.5):
        """Mark a specific step (default: last) as causing a backtrack."""
        if not self.current_episode_steps:
            return
        # If accessing via index -1, ensure list is not empty
        try:
            self.current_episode_steps[step_index].reward += penalty
        except IndexError:
            pass

    def finish_episode(self, final_reward: float):
        # Apply final reward to all steps
        for step in self.current_episode_steps:
            step.reward += final_reward
            
        if self.current_episode_steps:
             self.episode_buffer.append(self.current_episode_steps)
    
    def save_buffer(self):
        if not self.episode_buffer:
            return
            
        # Save a batch
        batch_id = str(uuid.uuid4())[:8]
        filename = f"batch_{batch_id}.pkl"
        path = os.path.join(self.save_dir, filename)
        
        try:
            with open(path, "wb") as f:
                pickle.dump(self.episode_buffer, f)
            # print(f"[Recorder] Saved {len(self.episode_buffer)} episodes to {filename}")
        except Exception as e:
            print(f"[Recorder] Failed to save batch: {e}")
            
        self.episode_buffer = []
