from typing import Optional  # Added Dict to typing

import torch
import torch.nn.functional as F

from src.util.struct import GateType


def calculate_consistency_loss(
    gate_types: torch.Tensor,  # [B, P, L]
    probs: torch.Tensor,  # [B, P, L, 2] - Gumbel-Softmax one-hot
    mask_valid: torch.Tensor,  # [B, P, L]
    device: torch.device,
) -> torch.Tensor:
    """Edge-level consistency loss using Gumbel-Softmax hard outputs.

    Uses the same discrete gradient path as calculate_full_logic_loss
    to avoid conflicting gradient signals in the optimizer.
    """
    p0 = probs[..., 0]
    p1 = probs[..., 1]

    # Context: A path is a sequence of gates.
    # v[i] is the output of v[i-1] (conceptually)?
    # NO. A path in reconvergent dataset is a sequence of nodes: n0 -> n1 -> n2 ...
    # where n1 is a fanout/fanin of n0?
    # Actually, in standard PODEM paths, it goes from input to output.
    # So v[i] is an input to v[i+1].
    # Wait, if v[i] is input to v[i+1], do we know the specific gate type connecting them?
    # Yes, gate_types[:, :, i+1] is the type of node i+1.
    # Node i is an input to Node i+1.

    # We check: Is v[i+1] consistent with v[i]?
    # Example: If Node i+1 is an AND gate, and Node i is 0, then Node i+1 MUST be 0.
    # If Node i+1 is an OR gate, and Node i is 1, then Node i+1 MUST be 1.
    # If Node i+1 is NOT, v[i+1] = 1 - v[i].

    prev_p0 = p0[:, :, :-1]
    prev_p1 = p1[:, :, :-1]

    cur_p0 = p0[:, :, 1:]
    cur_p1 = p1[:, :, 1:]

    gt_cur = gate_types[:, :, 1:]  # Type of the gate at i+1

    # Valid edges only
    valid_edges = mask_valid[:, :, 1:] & mask_valid[:, :, :-1]

    loss = torch.zeros_like(cur_p0)

    # --- 1. Controlling Input Violations (Dominant Inputs) ---

    # AND/NAND: Input 0 -> Output 0 (AND) / 1 (NAND)
    # Violation: Input is 0 (prev_p0 high), but Output is 1 (cur_p1 high) [for AND]
    is_and = gt_cur == GateType.AND
    if is_and.any():
        # If prev=0, cur MUST be 0. Disallow prev=0 & cur=1.
        # Penalty = P(prev=0) * P(cur=1)
        loss[is_and] += prev_p0[is_and] * cur_p1[is_and]

    is_nand = gt_cur == GateType.NAND
    if is_nand.any():
        # If prev=0, cur MUST be 1. Disallow prev=0 & cur=0.
        loss[is_nand] += prev_p0[is_nand] * cur_p0[is_nand]

    # OR/NOR: Input 1 -> Output 1 (OR) / 0 (NOR)
    # Violation: Input is 1 (prev_p1 high), but Output is 0 (cur_p0 high) [for OR]
    is_or = gt_cur == GateType.OR
    if is_or.any():
        loss[is_or] += prev_p1[is_or] * cur_p0[is_or]

    is_nor = gt_cur == GateType.NOR
    if is_nor.any():
        loss[is_nor] += prev_p1[is_nor] * cur_p1[is_nor]

    # --- 2. Deterministic Gate Violations (2x weight) ---

    is_not = gt_cur == GateType.NOT
    if is_not.any():
        loss[is_not] += 2.0 * (prev_p0[is_not] * cur_p0[is_not] + prev_p1[is_not] * cur_p1[is_not])

    is_buff = gt_cur == GateType.BUFF
    if is_buff.any():
        loss[is_buff] += 2.0 * (
            prev_p0[is_buff] * cur_p1[is_buff] + prev_p1[is_buff] * cur_p0[is_buff]
        )

    # --- 3. Non-Controlling Input Consistency (Weak) ---
    # AND: Input 1 -> Output can be 0 or 1 (depends on other inputs).
    # So if prev=1, we can't strictly constrain cur unless we know other inputs.
    # But current architecture only sees the *path*.
    # We ignore non-controlling cases in this partial view to avoid false penalties.
    # The "Reconvergence" interaction layer is supposed to resolve this via the global context,
    # but the *edge loss* must be physics-correct.
    # Penalizing AND(1, ?) -> ? is wrong.

    # Apply Mask
    loss = loss * valid_edges.float()

    # Averaging
    valid_count = valid_edges.sum().float().clamp(min=1.0)
    return loss.sum() / valid_count


def policy_loss_and_metrics(
    logits: torch.Tensor,
    node_ids: torch.Tensor,
    mask_valid: torch.Tensor,
    files: list[str],
    gate_types: torch.Tensor,
    # Constraints
    constraint_mask: torch.Tensor | None = None,
    constraint_vals: torch.Tensor | None = None,
    # Anchors
    anchor_p: torch.Tensor | None = None,
    anchor_l: torch.Tensor | None = None,
    anchor_v: torch.Tensor | None = None,
    solvability_logits: torch.Tensor | None = None,
    solvability_labels: torch.Tensor | None = None,
    anchor_alpha: float = 0.1,
    normalize_reward: bool = True,
    entropy_beta: float = 0.0,
    soft_edge_lambda: float = 1.0,
) -> tuple[torch.Tensor, float, float, float, float]:
    """Compute REINFORCE loss with LUT-inspired constraints and reconv consistency.

    Returns: (loss, avg_reward, valid_rate, edge_acc, constraint_violation_rate)
    """

    B, P, L, C = logits.shape


def calculate_logic_loss(
    node_ids: torch.Tensor,  # [B, P, L]
    gate_types: torch.Tensor,  # [B, P, L]
    probs: torch.Tensor,  # [B, P, L, 2] - Gumbel Softmax Outputs
    mask_valid: torch.Tensor,  # [B, P, L]
    device: torch.device,
) -> torch.Tensor:
    """
    Computes a Logic Consistency Loss by checking local gate logic.
    Vectorized implementation to avoid Python loops.
    """
    B, P, L_dim = node_ids.shape

    # 1. Identify Reconvergence Gate Indices per Batch Element
    # The reconvergence gate is the last valid node in each path.
    # Paths in a sample all reconverge to the same gate (by dataset construction).
    # We can pick the first valid path to find the location.

    path_lens = mask_valid.long().sum(dim=2)  # [B, P]
    # We need at least one path with length >= 2 (input -> reconv)
    valid_paths_mask = path_lens >= 2  # [B, P]

    # Filter batches that have no valid paths
    batch_has_valid = valid_paths_mask.any(dim=1)  # [B]
    if not batch_has_valid.any():
        return torch.tensor(0.0, device=device)

    # 2. Gather Inputs and Outputs
    # Inputs are at index (len-2), Output (Reconv) is at index (len-1)

    # We create a gather index for the last node (reconv) and 2nd last (input)
    # Indices: [B, P]
    last_idx = (path_lens - 1).clamp(min=0)
    second_last = (last_idx - 1).clamp(min=0)

    # Get Probabilities of logic-1 (Using Gumbel Softmax output passed in)
    probs_1 = probs[..., 1]  # [B, P, L]

    # Gather output probs: [B, P]
    # We gather from [B, P, L] using indices [B, P]
    idx_reconv = last_idx.unsqueeze(-1)  # [B, P, 1]
    p_out_per_path = probs_1.gather(2, idx_reconv).squeeze(-1)  # [B, P]

    # Gather input probs: [B, P]
    idx_input = second_last.unsqueeze(-1)  # [B, P, 1]
    p_in_per_path = probs_1.gather(2, idx_input).squeeze(-1)  # [B, P]

    # Gate Types for Reconv Gate: [B, P]
    # Should be identical across P for the same B (all paths reconverge to same gate)
    # We'll just take the mean/mode or assume consistency.
    gt_reconv = gate_types.gather(1, idx_reconv).squeeze(-1)  # [B, P]

    # 3. Compute Implied Probabilities (Vectorized)
    # We need to aggregate inputs for each batch sample.
    # Inputs are p_in_per_path[b, :] where valid_paths_mask[b, :] is True.

    # Since P is small (fixed max paths), we can mask and compute.
    # Mask invalid paths with identity values for the operation.
    # AND/NAND: identity = 1.0
    # OR/NOR: identity = 0.0

    # Prepare floating point mask
    f_mask = valid_paths_mask.float()

    # AND Logic: Product of inputs
    # Mask invalid entries with 1.0 (so they don't affect product)
    in_and = p_in_per_path * f_mask + (1.0 - f_mask)
    # Product across P dim
    implied_and = in_and.prod(dim=1)  # [B]

    # OR Logic: 1 - Product(1-inputs)
    # Mask invalid entries with 0.0 (so 1-0=1, product neutral)
    # Wait, for OR: we want prod(1-x). If specific x is invalid, we want it to be 0?
    # No, we want 1-x to be 1. So x must be 0.
    in_or = p_in_per_path * f_mask  # invalid -> 0
    implied_or = 1.0 - (1.0 - in_or).prod(dim=1)  # [B]

    # Reconv Gate Type (per batch)
    # We can take the max/first valid gate type per batch
    # (Assuming all paths in sample agree)
    # Gate types are integers. We need to index into operation results.

    # Let's get a representative gate type for each batch
    # We can pick the type from the first valid path.
    # Argmax of mask gives index of first True.
    first_valid_idx = valid_paths_mask.long().argmax(dim=1)  # [B]
    # Gather gate type
    batch_gt = gt_reconv.gather(1, first_valid_idx.unsqueeze(1)).squeeze(1)  # [B]

    # Calculate implied p1 for all types
    # Initialize with 0.5
    target_p1 = torch.full((B,), 0.5, device=device)

    is_and = batch_gt == GateType.AND
    is_nand = batch_gt == GateType.NAND
    is_or = batch_gt == GateType.OR
    is_nor = batch_gt == GateType.NOR
    is_not = batch_gt == GateType.NOT
    is_buff = batch_gt == GateType.BUFF
    is_xor = batch_gt == GateType.XOR
    is_xnor = batch_gt == GateType.XNOR

    # Assign Implied Targets
    # AND
    target_p1[is_and] = implied_and[is_and]

    # NAND (1 - AND)
    target_p1[is_nand] = 1.0 - implied_and[is_nand]

    # OR
    target_p1[is_or] = implied_or[is_or]

    # NOR (1 - OR)
    target_p1[is_nor] = 1.0 - implied_or[is_nor]

    # BUFF/NOT
    # For single input gates, we usually have 1 path, or duplicate paths.
    # Mean of inputs is a safe bet for duplicate paths.
    # But strictly, BUFF/NOT should have 1 input.
    # Sum / Count is better if we only have valid paths.
    sum_in = (p_in_per_path * f_mask).sum(dim=1)
    count_in = f_mask.sum(dim=1).clamp(min=1.0)
    mean_in = sum_in / count_in

    target_p1[is_buff] = mean_in[is_buff]
    target_p1[is_not] = 1.0 - mean_in[is_not]

    # XOR/XNOR Parity: target = 0.5 * (1 - prod(1 - 2*inputs))
    # Direct product is fine for small P (max paths per reconv)
    bits = (1.0 - 2.0 * p_in_per_path) * f_mask + (1.0 - f_mask)  # invalid -> 1.0
    parity_prod = bits.prod(dim=1)
    implied_xor = 0.5 * (1.0 - parity_prod)

    target_p1[is_xor] = implied_xor[is_xor]
    target_p1[is_xnor] = 1.0 - implied_xor[is_xnor]

    # 4. Compute Loss
    # Compare against Predicted Output
    # We assume 'mean' prediction across duplicate paths for the output as well
    sum_out = (p_out_per_path * f_mask).sum(dim=1)
    mean_out = sum_out / count_in

    # Only compute loss for supported gate types and valid batches
    supported = is_and | is_nand | is_or | is_nor | is_not | is_buff | is_xor | is_xnor
    mask_calc = batch_has_valid & supported

    if not mask_calc.any():
        return torch.tensor(0.0, device=device)

    loss = F.mse_loss(mean_out[mask_calc], target_p1[mask_calc])

    return loss


def calculate_full_logic_loss(
    gate_types: torch.Tensor,  # [B, P, L]
    probs: torch.Tensor,  # [B, P, L, 2]
    mask_valid: torch.Tensor,  # [B, P, L]
    device: torch.device,
) -> torch.Tensor:
    """
    Full-path gate consistency loss.
    Penalizes ALL edge-level gate logic violations along paths as a direct
    """
    probs_1 = probs[..., 1]  # [B, P, L]

    valid_edges = mask_valid[:, :, 1:] & mask_valid[:, :, :-1]  # [B, P, L-1]
    if not valid_edges.any():
        return torch.tensor(0.0, device=device)

    prev_p = probs_1[:, :, :-1]
    cur_p = probs_1[:, :, 1:]
    gt_cur = gate_types[:, :, 1:]  # [B, P, L-1]

    viol = torch.zeros_like(prev_p)

    # NOT: |cur - (1-prev)|^2
    m = gt_cur == GateType.NOT
    if m.any():
        viol[m] = 2.0 * ((cur_p[m] - (1.0 - prev_p[m])) ** 2)

    # BUFF: |cur - prev|^2
    m = gt_cur == GateType.BUFF
    if m.any():
        viol[m] = 2.0 * ((cur_p[m] - prev_p[m]) ** 2)

    # AND: cur <= prev => ReLU(cur - prev)^2
    m = gt_cur == GateType.AND
    if m.any():
        viol[m] = F.relu(cur_p[m] - prev_p[m]) ** 2

    # NAND: cur >= 1-prev => ReLU((1-prev) - cur)^2
    m = gt_cur == GateType.NAND
    if m.any():
        viol[m] = F.relu((1.0 - prev_p[m]) - cur_p[m]) ** 2

    # OR: cur >= prev => ReLU(prev - cur)^2
    m = gt_cur == GateType.OR
    if m.any():
        viol[m] = F.relu(prev_p[m] - cur_p[m]) ** 2

    # NOR: cur <= 1-prev => ReLU(cur - (1-prev))^2
    m = gt_cur == GateType.NOR
    if m.any():
        viol[m] = F.relu(cur_p[m] - (1.0 - prev_p[m])) ** 2

    viol = viol * valid_edges.float()

    valid_edge_counts = valid_edges.sum(dim=(1, 2)).float().clamp(min=1.0)
    loss_per_sample = viol.sum(dim=(1, 2)) / valid_edge_counts

    return loss_per_sample.mean()


def reinforce_loss(
    logits: torch.Tensor,
    gate_types: torch.Tensor,
    mask_valid: torch.Tensor,
    solvability_logits: Optional[torch.Tensor] = None,
    solvability_labels: Optional[torch.Tensor] = None,
    anchor_p: Optional[torch.Tensor] = None,
    anchor_l: Optional[torch.Tensor] = None,
    anchor_v: Optional[torch.Tensor] = None,
    entropy_beta: float = 0.01,
    constraint_mask: Optional[torch.Tensor] = None,
    constraint_vals: Optional[torch.Tensor] = None,
    node_ids: Optional[torch.Tensor] = None,
    lambda_logic: float = 0.0,
    lambda_full_logic: float = 0.0,
    soft_edge_lambda: float = 1.0,
    normalize_reward: bool = True,
    anchor_alpha: float = 0.1,
    gumbel_temp: float = 1.0,
) -> tuple[torch.Tensor, float, float, float, float]:
    """Compute Loss using Gumbel-Softmax for differentiable logic consistency.

    The 'reward' concepts from RL are kept for metric logging but removed from the
    gradient path in favor of direct logic differentiation.

    Returns: (loss, avg_reward, valid_rate, edge_acc, constraint_violation_rate)
    """

    B, P, L, C = logits.shape

    # Gumbel Softmax Action Sampling (Straight-Through Estimator)
    # hard=True:
    #   Forward pass: one-hot discrete values (0 or 1)
    #   Backward pass: uses gradients of the Gumbel-Softmax distribution
    # We use this ONE-HOT output for all logic calculations to propagate gradients.
    actions_one_hot = F.gumbel_softmax(logits, tau=gumbel_temp, hard=True, dim=-1)  # [B, P, L, 2]

    # Extract indices for legacy code (constraint masking, metrics)
    actions = actions_one_hot.argmax(
        dim=-1
    )  # [B, P, L] - No grad flow through argmax usually, but OK here as we use one_hot for loss

    # constraint_metrics
    constraint_loss = torch.tensor(0.0, device=logits.device)
    constraint_violation_rate = torch.tensor(0.0, device=logits.device)

    # Enforce constraints if provided
    if constraint_mask is not None and constraint_vals is not None:
        if constraint_mask.any():
            valid_constraints = constraint_mask.view(-1)
            flat_logits = logits.view(-1, 2)
            flat_targets = constraint_vals.view(-1)

            # CE Loss on constrained nodes (Supervised)
            c_loss = F.cross_entropy(
                flat_logits[valid_constraints], flat_targets[valid_constraints]
            )
            constraint_loss = c_loss * 1.0

            # Metric: Violation rate
            preds = actions[constraint_mask]
            targets = constraint_vals[constraint_mask]
            violations = (preds != targets).float().sum()
            total_c = targets.numel()
            constraint_violation_rate = torch.tensor(
                violations.item() / max(1, total_c), device=logits.device
            )

            # Forcing: Update actions to match constraints to ensure downstream
            # path consistency?
            # In RL we could overwrite actions. In Gumbel matching loss is usually enough.
            # But to help convergence we can overwrite the 'hard' actions for the
            # forward pass context if we wanted. But Gumbel hard=True makes it discrete.
            # Let's overwrite `actions` indices for metrics, but we can't easily
            # overwrite `actions_one_hot` without breaking gradient flow unless we
            # are careful.
            # Since constraint_loss is strong, we'll assume it converges.
            # However, existing code overwrote actions.
            actions[constraint_mask] = constraint_vals[constraint_mask]

    # Solvability Loss
    solvability_loss = torch.tensor(0.0, device=logits.device)
    if solvability_logits is not None and solvability_labels is not None:
        weights = torch.tensor([10.0, 1.0], device=logits.device)
        solvability_loss = (
            F.cross_entropy(solvability_logits, solvability_labels, weight=weights) * 1.0
        )

    # RL Reward Logic (For Logging Only)
    unsat_reward = torch.zeros(B, dtype=torch.float32, device=logits.device)

    if solvability_labels is not None:
        unsat_mask = solvability_labels == 1

        if unsat_mask.any():
            pred_solv = torch.argmax(solvability_logits, dim=-1)
            correct = (pred_solv[unsat_mask] == 1).float()
            unsat_reward[unsat_mask] = torch.where(
                correct == 1, torch.ones_like(correct), torch.full_like(correct, -1.0)
            )

    # Accumulate loss terms in a list for clean gradient flow
    loss_terms: list[torch.Tensor] = []

    # Vectorized Logic Consistency (Metrics Only)
    # We use `actions` (indices) for metrics to see "Hard" violations
    valid_edges = mask_valid[:, :, 1:] & mask_valid[:, :, :-1]  # [B, P, L-1]
    prev_vals = actions[:, :, :-1]
    cur_vals = actions[:, :, 1:]
    gt_cur = gate_types[:, :, 1:]

    edge_ok = torch.ones_like(prev_vals, dtype=torch.bool)
    m = gt_cur == GateType.NOT
    edge_ok[m] &= cur_vals[m] == (1 - prev_vals[m])
    m = gt_cur == GateType.BUFF
    edge_ok[m] &= cur_vals[m] == prev_vals[m]
    m = gt_cur == GateType.AND
    edge_ok[m] &= cur_vals[m] <= prev_vals[m]
    m = gt_cur == GateType.NAND
    edge_ok[m] &= cur_vals[m] >= (1 - prev_vals[m])
    m = gt_cur == GateType.OR
    edge_ok[m] &= cur_vals[m] >= prev_vals[m]
    m = gt_cur == GateType.NOR
    edge_ok[m] &= cur_vals[m] <= (1 - prev_vals[m])

    wrong_edges = (~edge_ok) & valid_edges
    local_wrong = wrong_edges.sum(dim=(1, 2))
    checked = valid_edges.sum(dim=(1, 2))
    edge_wrong_sum = local_wrong.sum().item()
    edge_total_sum = checked.sum().item()

    # Vectorized Reconvergence Failures (Metrics Only)
    path_len = mask_valid.long().sum(dim=-1)
    last_idx = (path_len - 1).clamp(min=0)
    last_idx_exp = last_idx.unsqueeze(-1)
    last_vals = actions.gather(2, last_idx_exp).squeeze(-1)
    path_valid_mask = path_len > 0

    neg_inf = -999.0
    pos_inf = 999.0
    lv_float = last_vals.float()
    vm_float = path_valid_mask.float()
    max_v = (lv_float * vm_float + neg_inf * (1 - vm_float)).max(dim=-1).values
    min_v = (lv_float * vm_float + pos_inf * (1 - vm_float)).min(dim=-1).values

    has_valid_paths = path_valid_mask.sum(dim=-1) > 0
    reconv_fail_mask = (min_v < max_v) & has_valid_paths
    reconv_wrong = reconv_fail_mask.float()

    # Metrics
    with torch.no_grad():
        # Granular path-wise accuracy (instead of strict sample-wise validity)
        path_wrong = wrong_edges.sum(dim=2)  # [B, P]
        path_valid = (path_wrong == 0) & path_valid_mask  # [B, P]

        # Simplify by directly removing the solvable masking logic
        total_paths = path_valid_mask.float().sum().item()
        total_valid_paths = path_valid.float().sum().item()

        # Fallback to local error for reward shaping
        local_err = local_wrong.float()

        # Path Accuracy (percentage of paths entirely consistent)
        valid_rate = torch.tensor(total_valid_paths / max(1.0, total_paths), device=logits.device)
        edge_acc = torch.tensor(
            (edge_total_sum - edge_wrong_sum) / max(1.0, edge_total_sum), device=logits.device
        )

        # Calculate a pseudo-reward for logging consistency
        local_reward_shaping = (1.0 - (local_err / checked.clamp(min=1.0))) * 2.0 - 1.0
        reconv_bonus = 0.5
        reconv_penalty = -2.0
        sat_base_reward = torch.where(
            reconv_wrong == 0,
            local_reward_shaping + reconv_bonus,
            torch.min(local_reward_shaping, torch.tensor(reconv_penalty, device=logits.device)),
        )
        reward = sat_base_reward.clone()
        avg_reward = reward.mean()

    # -----------------------------------------------------------------------
    # Differentiable Losses
    # -----------------------------------------------------------------------

    # 1. Reconvergence Consistency Loss (MSE on Logits / Gumbel Probs)
    # We use logits or gumbel outputs. Gumbel outputs are safer for consistency.
    # Gather output probs [B, P, 2]
    # last_idx_logits = last_idx_exp.unsqueeze(-1).expand(actions.size(0), actions.size(1), 1, 2)
    # last_probs = actions_one_hot.gather(2, last_idx_logits).squeeze(2) # [B, P, 2]
    # Actually, let's use the code from before but on LOGITS to permit gradients
    # even without Gumbel hard path if needed, BUT mixing them is tricky.
    # Let's use logits for the MSE consistency as it provides a smooth signal.

    # Gather logits at last_idx: [B, P, C]
    last_idx_logits = last_idx_exp.unsqueeze(-1).expand(
        actions.size(0), actions.size(1), 1, logits.shape[-1]
    )
    last_logits = logits.gather(2, last_idx_logits).squeeze(2)  # [B, P, C]

    mask_exp = path_valid_mask.unsqueeze(-1)  # [B, P, 1]
    count_paths = mask_exp.sum(dim=1).clamp(min=1.0)  # [B, 1]
    sum_logits = (last_logits * mask_exp).sum(dim=1)  # [B, C]
    mean_logits = sum_logits / count_paths  # [B, C]

    diff = last_logits - mean_logits.unsqueeze(1)
    mse_per_sample = ((diff**2) * mask_exp).sum(dim=(1, 2)) / count_paths.squeeze(-1)  # [B]
    loss_terms.append(0.5 * mse_per_sample.mean())

    # 2. Vectorized Soft Edge Loss (Using Gumbel Outputs one_hot)
    # This acts as a differentiable consistency check.
    probs_1 = actions_one_hot[..., 1]  # [B, P, L]
    prev_p = probs_1[:, :, :-1]
    cur_p = probs_1[:, :, 1:]

    viol = torch.zeros_like(prev_p)
    # NOT: |cur - (1-prev)|^2
    m = gt_cur == GateType.NOT
    viol[m] = (cur_p[m] - (1.0 - prev_p[m])) ** 2
    # BUFF: |cur - prev|^2
    m = gt_cur == GateType.BUFF
    viol[m] = (cur_p[m] - prev_p[m]) ** 2
    # AND: cur <= prev => ReLU(cur - prev)^2
    m = gt_cur == GateType.AND
    viol[m] = F.relu(cur_p[m] - prev_p[m]) ** 2
    # NAND: cur >= 1-prev => ReLU((1-prev) - cur)^2
    m = gt_cur == GateType.NAND
    viol[m] = F.relu((1.0 - prev_p[m]) - cur_p[m]) ** 2
    # OR: cur >= prev => ReLU(prev - cur)^2
    m = gt_cur == GateType.OR
    viol[m] = F.relu(prev_p[m] - cur_p[m]) ** 2
    # NOR: cur <= 1-prev => ReLU(cur - (1-prev))^2
    m = gt_cur == GateType.NOR
    viol[m] = F.relu(cur_p[m] - (1.0 - prev_p[m])) ** 2

    viol = viol * valid_edges.float()

    valid_edge_counts = valid_edges.sum(dim=(1, 2)).float().clamp(min=1.0)
    edge_loss_per_sample = viol.sum(dim=(1, 2)) / valid_edge_counts

    if soft_edge_lambda > 0:
        loss_terms.append(float(soft_edge_lambda) * edge_loss_per_sample.mean())

    # 3. Entropy regularization (Using logits)
    if entropy_beta > 0.0:
        probs = torch.softmax(logits, dim=-1)
        ent = -(probs * torch.log_softmax(logits, dim=-1)).sum(dim=-1)
        ent_mean = (ent * mask_valid.float()).sum() / torch.clamp(mask_valid.float().sum(), min=1.0)
        loss_terms.append(-float(entropy_beta) * ent_mean)

    # 4. Anchor Supervision (Standard CE)
    if anchor_p is not None and anchor_l is not None and anchor_v is not None:
        valid_anchors = (anchor_p >= 0) & (anchor_l >= 0)
        if solvability_labels is not None:
            valid_anchors = valid_anchors & (solvability_labels == 0)
        if valid_anchors.any():
            b_idx = torch.arange(logits.size(0), device=logits.device)[valid_anchors]
            p_idx = anchor_p[valid_anchors]
            l_idx = anchor_l[valid_anchors]
            targets = anchor_v[valid_anchors].long().clamp(0, 1)
            pred_logits = logits[b_idx, p_idx, l_idx]
            sup_loss = F.cross_entropy(pred_logits, targets)
            loss_terms.append(sup_loss)

    # 5. Add accumulated auxiliary losses
    if constraint_loss.requires_grad or constraint_loss.item() > 0:
        loss_terms.append(constraint_loss)
    if solvability_loss.requires_grad or solvability_loss.item() > 0:
        loss_terms.append(solvability_loss)

    # 6. Reconvergence Logic Loss (Differentiable via Gumbel)
    if lambda_logic > 0.0:
        logic_loss = calculate_consistency_loss(
            gate_types=gate_types,
            probs=actions_one_hot,
            mask_valid=mask_valid,
            device=logits.device,
        )
        loss_terms.append(lambda_logic * logic_loss)

    # 7. Full-Path Gate Consistency Loss
    if lambda_full_logic > 0.0:
        full_logic_loss = calculate_full_logic_loss(
            gate_types=gate_types,
            probs=actions_one_hot,
            mask_valid=mask_valid,
            device=logits.device,
        )
        loss_terms.append(lambda_full_logic * full_logic_loss)

    # Sum all loss terms
    if loss_terms:
        loss = torch.stack(loss_terms).sum()
    else:
        loss = torch.tensor(0.0, device=logits.device)

    return loss, avg_reward, valid_rate, edge_acc, constraint_violation_rate


def _debug_metrics_from_logits(
    logits: torch.Tensor,
    node_ids: torch.Tensor,
    mask_valid: torch.Tensor,
    files: list[str],
    anchor_p: Optional[torch.Tensor] = None,
    anchor_l: Optional[torch.Tensor] = None,
    anchor_v: Optional[torch.Tensor] = None,
    solvability_logits: Optional[torch.Tensor] = None,
    solvability_labels: Optional[torch.Tensor] = None,
) -> dict:
    """
    Granular metrics for debugging, specifically reconvergence consistency
    and solvability accuracy.
    """
    with torch.no_grad():
        actions = logits.argmax(dim=-1)
        B, P, _ = actions.shape

        # Solvability Accuracy
        solv_acc = 0.0
        if solvability_logits is not None and solvability_labels is not None:
            pred_solv = solvability_logits.argmax(dim=-1)
            solv_acc = (pred_solv == solvability_labels).float().mean().item()

        # Reconvergence Consistency (Do all paths in a sample agree on the terminal value?)
        path_len = mask_valid.long().sum(dim=-1)
        last_idx = (path_len - 1).clamp(min=0)
        last_idx_exp = last_idx.unsqueeze(-1)
        last_vals = actions.gather(2, last_idx_exp).squeeze(-1)  # [B, P]

        path_valid_mask = path_len > 0
        match_count = 0
        total_reconv_samples = 0

        for b in range(B):
            valid_paths = last_vals[b][path_valid_mask[b]]
            if len(valid_paths) > 1:
                total_reconv_samples += 1
                if (valid_paths == valid_paths[0]).all():
                    match_count += 1

        reconv_match = match_count / max(1, total_reconv_samples)

        return {
            "edge_acc": 1.0,  # Placeholder as we don't have gate_types
            "reconv_match_rate": reconv_match,
            "solvability_acc": solv_acc,
            "edges_per_sample": 0.0,
        }
