import sys
import os
import faulthandler

faulthandler.enable()

sys.path.append(os.getcwd())
from src.atpg.ai_podem import ai_podem, ModelPairPredictor, HierarchicalReconvSolver, AiPodemConfig
from src.util.io import parse_bench_file
from src.atpg.podem import get_all_faults

circuit, tg = parse_bench_file("data/bench/ISCAS85/c432.bench")
faults = get_all_faults(circuit, tg)

cfg = AiPodemConfig(
    model_path="checkpoints/unlinked_candidate/best_model.pth",
    device="cpu",
    enable_ai_activation=True,
    enable_ai_propagation=True,
)

predictor = ModelPairPredictor(circuit, "data/bench/ISCAS85/c432.bench", cfg)
solver = HierarchicalReconvSolver(circuit, predictor)

for f in faults:
    try:
        print(f"Trying fault {f.gate_id}", flush=True)
        ai_podem(
            circuit,
            f,
            tg,
            model_path="checkpoints/unlinked_candidate/best_model.pth",
            circuit_path="data/bench/ISCAS85/c432.bench",
            enable_ai_activation=True,
            enable_ai_propagation=True,
            predictor=predictor,
            solver=solver,
        )
    except Exception as e:
        print(f"Error {e}")
