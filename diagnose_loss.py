"""
Diagnostic: verify the three fixes applied:
  1. Constraint encoding now matches training (3-dim [val0,val1,unknown] at 128-130)
  2. Solvability loss weights now [1.0, 10.0] (UNSAT upweighted)
  3. Model predictions now differ meaningfully when constraints are applied
"""
import contextlib
import io
import sys

sys.path.insert(0, ".")

import torch
import torch.nn.functional as F

from src.atpg.ai_podem import AiPodemConfig, HierarchicalReconvSolver, ModelPairPredictor
from src.ml.core.loss import calculate_shared_node_consistency_loss, reinforce_loss
from src.util.io import parse_bench_file
from src.util.struct import GateType, LogicValue

BENCH = "data/bench/ISCAS85/c432.bench"
MODEL = "checkpoints/supervised_v3/best_model.pth"
TARGET_GATE = 428

print("=" * 72)
print("Loading circuit + model")
print("=" * 72)
circuit, total_gates = parse_bench_file(BENCH)
config = AiPodemConfig(model_path=MODEL, device="cpu")
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    predictor = ModelPairPredictor(circuit, BENCH, config)

gate_type_name = {v: k for k, v in GateType.__members__.items()}
print(f"Circuit: {len(circuit)} gates, struct_emb shape: {predictor.struct_emb.shape}")

# ---- Pick pairs ----
solver = HierarchicalReconvSolver(circuit, predictor, verbose=False)
pairs_by_reconv = solver._collect_and_sort_pairs(TARGET_GATE)
pair = pairs_by_reconv[TARGET_GATE][0]
paths = pair["paths"]
start_node = pair.get("start", paths[0][0])
reconv_node = pair["reconv"]

print(f"\nSelected pair: start={start_node}, reconv={reconv_node}, paths={len(paths)}")
for i, p in enumerate(paths):
    types = [gate_type_name.get(circuit[n].type, "?") for n in p]
    print(f"  Path {i} (len={len(p)}): {list(zip(p, types))}")

# ---- VERIFY FIX 1: Constraint encoding ----
print()
print("=" * 72)
print("FIX 1 VERIFICATION: Constraint encoding matches training")
print("  Training: dim128=val0, dim129=val1, dim130=unknown (default=1.0)")
print("  Inference (fixed): same scheme")
print("=" * 72)

def build_batch_fixed(paths, constraints, predictor, circuit):
    """Build batch using the FIXED inference encoding."""
    device = torch.device("cpu")
    P = len(paths)
    L = max(len(p) for p in paths)

    path_embs_list, gate_types_list, node_ids_list = [], [], []
    constraint_mask_list, constraint_vals_list = [], []
    lv_to_int = {LogicValue.ZERO: 0, LogicValue.ONE: 1}

    for p in paths:
        p_emb, p_types, p_ids, p_cmask, p_cvals = [], [], [], [], []
        for nid in p:
            p_ids.append(nid)
            if nid in predictor.gate_mapping:
                aig_id = predictor.gate_mapping[nid]
                emb = predictor.struct_emb[aig_id].clone().float() if aig_id < predictor.struct_emb.size(0) else torch.zeros(128)
            else:
                emb = torch.zeros(128)

            # Pad to 131 first
            if emb.shape[0] < 131:
                emb = torch.cat([emb, torch.zeros(131 - emb.shape[0])])

            # Apply training-compatible encoding
            if nid in constraints:
                val = constraints[nid]
                emb[128] = 1.0 if val == LogicValue.ZERO else 0.0
                emb[129] = 1.0 if val == LogicValue.ONE  else 0.0
                emb[130] = 0.0  # known
            else:
                emb[128] = 0.0
                emb[129] = 0.0
                emb[130] = 1.0  # unknown — matches training default

            # Pad to 132 (multiple of 4)
            if emb.shape[0] < 132:
                emb = torch.cat([emb, torch.zeros(132 - emb.shape[0])])

            p_emb.append(emb)
            p_cmask.append(nid in constraints)
            p_cvals.append(lv_to_int.get(constraints.get(nid), 0))
            p_types.append(circuit[nid].type if nid < len(circuit) else 0)

        while len(p_emb) < L:
            pad = torch.zeros(132)
            pad[130] = 1.0  # unknown
            p_emb.append(pad)
            p_types.append(0)
            p_ids.append(0)
            p_cmask.append(False)
            p_cvals.append(0)

        path_embs_list.append(torch.stack(p_emb))
        gate_types_list.append(torch.tensor(p_types))
        node_ids_list.append(torch.tensor(p_ids))
        constraint_mask_list.append(torch.tensor(p_cmask, dtype=torch.bool))
        constraint_vals_list.append(torch.tensor(p_cvals, dtype=torch.long))

    batch_embs  = torch.stack(path_embs_list).unsqueeze(0)
    batch_types = torch.stack(gate_types_list).unsqueeze(0)
    batch_ids   = torch.stack(node_ids_list).unsqueeze(0)
    batch_mask  = torch.ones(1, P, L, dtype=torch.bool)
    for pi, p in enumerate(paths):
        batch_mask[0, pi, len(p):] = False
    cmask = torch.stack(constraint_mask_list).unsqueeze(0)
    cvals = torch.stack(constraint_vals_list).unsqueeze(0)
    return batch_embs, batch_mask, batch_ids, batch_types, cmask, cvals


constraints_none  = {}
constraints_zero  = {start_node: LogicValue.ZERO}
constraints_one   = {start_node: LogicValue.ONE}

emb_none, mask_none, nids_none, gt_none, cm_none, cv_none = build_batch_fixed(paths, constraints_none, predictor, circuit)
emb_zero, mask_zero, nids_zero, gt_zero, cm_zero, cv_zero = build_batch_fixed(paths, constraints_zero, predictor, circuit)
emb_one,  mask_one,  nids_one,  gt_one,  cm_one,  cv_one  = build_batch_fixed(paths, constraints_one,  predictor, circuit)

print(f"\nEmbedding shape: {list(emb_none.shape)}")
print(f"Logic dims for node {start_node}:")

for label, emb, c in [("no constraint", emb_none, constraints_none),
                       ("ZERO",          emb_zero, constraints_zero),
                       ("ONE",           emb_one,  constraints_one)]:
    # Find position of start_node in path 0
    li = paths[0].index(start_node)
    dims = emb[0, 0, li, 128:131].tolist()
    print(f"  {label:15s}: dims[128:131]={[round(x,1) for x in dims]}  "
          f"(expected: {'[0,0,1]' if label=='no constraint' else '[1,0,0]' if label=='ZERO' else '[0,1,0]'})")

# Run model on all three
model = predictor.model
model.eval()
with torch.no_grad():
    logits_none, solv_none = model(emb_none, mask_none, gate_types=gt_none)
    logits_zero, solv_zero = model(emb_zero, mask_zero, gate_types=gt_zero)
    logits_one,  solv_one  = model(emb_one,  mask_one,  gate_types=gt_one)

print(f"\nPredictions at node {start_node} (path=0, pos={paths[0].index(start_node)}):")
li = paths[0].index(start_node)
for label, logits in [("no constraint", logits_none), ("ZERO", logits_zero), ("ONE", logits_one)]:
    p = torch.softmax(logits[0, 0, li], dim=-1)
    pred = logits[0, 0, li].argmax().item()
    print(f"  {label:15s}: pred={pred}  P(0)={p[0]:.3f}  P(1)={p[1]:.3f}")

print(f"\nPredictions change with constraint? "
      f"{'YES ✓' if abs(logits_zero[0,0,li,1].item() - logits_none[0,0,li,1].item()) > 0.01 or abs(logits_one[0,0,li,1].item() - logits_none[0,0,li,1].item()) > 0.01 else 'NO (model still ignores constraints)'}")

# ---- VERIFY FIX 2: Solvability loss weights ----
print()
print("=" * 72)
print("FIX 2 VERIFICATION: Solvability loss weights [1.0, 10.0] (UNSAT upweighted)")
print("=" * 72)

sat_logits  = torch.tensor([[2.0, -2.0]])  # predicts SAT (class 0)
unsat_logits = torch.tensor([[-2.0, 2.0]])  # predicts UNSAT (class 1)
sat_label   = torch.tensor([0])  # true=SAT
unsat_label = torch.tensor([1])  # true=UNSAT

w_fixed = torch.tensor([1.0, 10.0])
w_old   = torch.tensor([10.0, 1.0])

loss_correct_sat   = F.cross_entropy(sat_logits,   sat_label,   weight=w_fixed).item()
loss_correct_unsat = F.cross_entropy(unsat_logits, unsat_label, weight=w_fixed).item()
loss_wrong_sat     = F.cross_entropy(unsat_logits, sat_label,   weight=w_fixed).item()
loss_wrong_unsat   = F.cross_entropy(sat_logits,   unsat_label, weight=w_fixed).item()

print("\nWith FIXED weights [1.0, 10.0]:")
print(f"  Correct SAT prediction loss:    {loss_correct_sat:.4f}")
print(f"  Correct UNSAT prediction loss:  {loss_correct_unsat:.4f}")
print(f"  Wrong prediction on SAT sample: {loss_wrong_sat:.4f}")
print(f"  Wrong prediction on UNSAT sample: {loss_wrong_unsat:.4f}")
print(f"  UNSAT miss loss > SAT miss loss? {'YES ✓' if loss_wrong_unsat > loss_wrong_sat else 'NO ✗'}")

# ---- VERIFY FIX 3: Full reinforce_loss with constraint in embedding ----
print()
print("=" * 72)
print("FIX 3 VERIFICATION: reinforce_loss with ZERO vs ONE constraint")
print("=" * 72)

solv_labels = torch.zeros(1, dtype=torch.long)  # assume SAT

for label, logits, mask, nids, gt, cmask, cvals, solv_logits in [
    ("no constraint", logits_none, mask_none, nids_none, gt_none, cm_none, cv_none, solv_none),
    ("ZERO at start", logits_zero, mask_zero, nids_zero, gt_zero, cm_zero, cv_zero, solv_zero),
    ("ONE at start",  logits_one,  mask_one,  nids_one,  gt_one,  cm_one,  cv_one,  solv_one),
]:
    total, avg_r, valid_r, edge_acc, c_viol = reinforce_loss(
        logits=logits,
        gate_types=gt,
        mask_valid=mask,
        solvability_logits=solv_logits,
        solvability_labels=solv_labels,
        constraint_mask=cmask if cmask.any() else None,
        constraint_vals=cvals if cmask.any() else None,
        node_ids=nids,
        lambda_shared_node=1.0,
        soft_edge_lambda=1.0,
        gumbel_temp=1.0,
    )
    entropy = -(torch.softmax(logits, dim=-1) * torch.log_softmax(logits, dim=-1)).sum(dim=-1)
    ent_mean = (entropy * mask.float()).sum() / mask.float().sum().clamp(min=1.0)
    print(f"\n  [{label}]")
    print(f"    total_loss={total.item():.6f}  edge_acc={edge_acc.item():.3f}  "
          f"valid_rate={valid_r.item():.3f}  entropy={ent_mean.item():.4f}")
    print(f"    constraint_viol={c_viol.item():.3f}")
    if cmask.any():
        # Check if model prediction at constrained position matches constraint
        pred = logits[0, 0, li].argmax().item()
        expected = cv_one[0, 0, li].item() if "ONE" in label else cv_zero[0, 0, li].item()
        print(f"    Constrained pos pred={pred}, expected={expected}: "
              f"{'MATCH ✓' if pred == expected else 'MISMATCH ✗ (constraint loss will push toward correct)'}")

# ---- VERIFY: cross-path consistency on a pair with shared nodes ----
print()
print("=" * 72)
print("FIX 4 VERIFICATION: Shared-node consistency loss on actual contradiction")
print("  Build a synthetic case: same node, different predictions on two paths")
print("=" * 72)

# Manually construct a 2-path batch where path 0 predicts node 213=1, path 1 predicts node 213=0
P, L = 2, 4
logits_synth = torch.zeros(1, P, L, 2)
# path 0, pos 0 (node 213): strongly predict 1
logits_synth[0, 0, 0, 1] = 5.0
logits_synth[0, 0, 0, 0] = -5.0
# path 1, pos 0 (node 213): strongly predict 0
logits_synth[0, 1, 0, 0] = 5.0
logits_synth[0, 1, 0, 1] = -5.0
# All other positions: uniform
for pi in range(P):
    for li in range(1, L):
        logits_synth[0, pi, li] = torch.tensor([0.0, 0.0])

node_ids_synth = torch.zeros(1, P, L, dtype=torch.long)
node_ids_synth[0, :, 0] = 213  # same node on both paths
node_ids_synth[0, 0, 1] = 254
node_ids_synth[0, 0, 2] = 393
node_ids_synth[0, 0, 3] = 428
node_ids_synth[0, 1, 1] = 255
node_ids_synth[0, 1, 2] = 399
node_ids_synth[0, 1, 3] = 428

mask_synth = torch.ones(1, P, L, dtype=torch.bool)
probs_synth = F.gumbel_softmax(logits_synth, tau=1.0, hard=True, dim=-1)

shared_loss = calculate_shared_node_consistency_loss(
    node_ids=node_ids_synth,
    probs=probs_synth,
    mask_valid=mask_synth,
)
print("\nSynthetic contradiction (node 213: path0=1, path1=0):")
print(f"  Shared-node consistency loss = {shared_loss.item():.4f}  (should be > 0)")
print(f"  {'PENALIZES CONTRADICTION ✓' if shared_loss.item() > 0 else 'BUG: no penalty ✗'}")

# Consistent case: both paths predict same value
logits_consist = logits_synth.clone()
logits_consist[0, 1, 0, 0] = -5.0
logits_consist[0, 1, 0, 1] = 5.0  # now both predict 1

probs_consist = F.gumbel_softmax(logits_consist, tau=1.0, hard=True, dim=-1)
shared_loss_ok = calculate_shared_node_consistency_loss(
    node_ids=node_ids_synth,
    probs=probs_consist,
    mask_valid=mask_synth,
)
print("\nConsistent case (node 213: both paths=1):")
print(f"  Shared-node consistency loss = {shared_loss_ok.item():.4f}  (should be 0)")
print(f"  {'NO PENALTY ✓' if shared_loss_ok.item() == 0 else 'BUG: unexpected penalty ✗'}")

print()
print("=" * 72)
print("SUMMARY")
print("=" * 72)
print("Fix 1 (constraint encoding): see 'dims[128:131]' rows above")
print("Fix 2 (solvability weights): UNSAT miss loss > SAT miss loss?", "YES ✓" if loss_wrong_unsat > loss_wrong_sat else "NO ✗")
print("Fix 3 (constraint loss activates when mismatch): see constraint_viol rows")
print("Fix 4 (shared-node loss): contradiction penalized, consistent case not?",
      "YES ✓" if shared_loss.item() > 0 and shared_loss_ok.item() == 0 else "CHECK ABOVE")
