"""
PODEM (Path-Oriented Decision Making) Algorithm Implementation
MODIFIED to support MLP PPO (stage1), GCN PPO (stage2), and GCN PPO + RND (stage3).
"""

import sys
import time
from collections import defaultdict
from typing import (  # ...updated (added Optional, Any, Callable)
    Callable,
    List,
    Optional,
)

import src.atpg.logic_sim_three as logic_sim
from src.atpg.logic_sim_three import (
    d_frontier,
    fault_is_at_po,
    print_pi,
    set_d_frontier_sort,
)
from src.atpg.scoap import calculate_scoap
from src.atpg.util import (
    calculate_distance_to_primary_inputs,
    calculate_distance_to_primary_outputs,
    get_x_fanin,
)
from src.util.io import (
    get_gate_type_str as get_gate_type_name,
)
from src.util.io import (
    parse_bench_file,
)
from src.util.struct import (
    Fault,
    Gate,
    GateType,
    LogicValue,
)

# Set maximum recursion depth
sys.setrecursionlimit(500000)

# PODEM constants
# PODEM constants
UNTESTABLE = 0
SUCCESS = 1
TIMEOUT = 2
BACKTRACK_LIMIT = 3

# Global variables
depth = 0
scoap_calculated = False
rl_agent = None  # Global RL agent instance (None => no RL)
# Distances to primary inputs (backward) and to primary outputs (forward)
gate_distances_back = {}  # shortest distance from each gate to any primary input
gate_distances_fwd = {}  # shortest distance from each gate to any primary output
rl_env = None  # Persistent RL environment for RL backtrace
current_agent_type = None  # Retained for compatibility / logging
VERBOSE = False
rl_budget = float("inf")  # No limit on RL backtrace uses per fault

# When True, print backtrace/backtrack decisions during search
TRACE_DECISIONS = False

# Statistics tracking variables
backtrack_count = 0
backtrace_count = 0
total_recursive_calls = 0
backtrace_hops = 0

rl_calls = 0
rl_choices_used = 0
rl_calls_by_type = defaultdict(int)

TRAIN_BT_BUDGET = None

backtrace_function = None

# Time-based timeout
podem_start_time = 0
podem_timeout = float("inf")
topological_order = []


def get_rl_usage_counters():
    return {
        "rl_calls": rl_calls,
        "rl_choices_used": rl_choices_used,
        "rl_calls_by_type": dict(rl_calls_by_type),
    }


def reset_statistics():
    """Reset global statistics counters"""
    global backtrack_count, backtrace_count, total_recursive_calls
    backtrack_count = 0
    backtrace_count = 0
    total_recursive_calls = 0


def get_statistics():
    """Get current statistics as a dictionary"""
    global backtrack_count, backtrace_count, total_recursive_calls, backtrace_hops
    return {
        "backtrack_count": backtrack_count,
        "backtrace_count": backtrace_count,
        "backtrace_hops": backtrace_hops,
        "total_recursive_calls": total_recursive_calls,
    }


def set_trace_decisions(enabled: bool = True) -> None:
    """Enable/disable decision tracing prints during PODEM search."""
    global TRACE_DECISIONS
    TRACE_DECISIONS = bool(enabled)


def _trace(msg: str) -> None:
    if TRACE_DECISIONS:
        print(msg)


def initialize(circuit: List[Gate], total_gates: int):
    global gate_distances_back, gate_distances_fwd

    gate_distances_back = calculate_distance_to_primary_inputs(circuit, total_gates)
    gate_distances_fwd = calculate_distance_to_primary_outputs(circuit, total_gates)

    # Reset all gate values to X to prevent state pollution from previous runs
    for i in range(1, total_gates + 1):
        circuit[i].val = LogicValue.XD


def podem(
    circuit: List[Gate],
    fault: Fault,
    total_gates: int,
    backtrace_func: Optional[Callable] = None,
    timeout: float = float("inf"),
    max_backtracks: int = 100000,
) -> bool:
    """Entry point for PODEM algorithm."""
    global \
        depth, \
        scoap_calculated, \
        rl_agent, \
        current_agent_type, \
        rl_env, \
        rl_budget, \
        gate_distances_back, \
        gate_distances_fwd, \
        backtrace_function, \
        podem_start_time, \
        podem_timeout, \
        podem_max_backtracks, \
        topological_order

    podem_start_time = time.time()
    podem_timeout = timeout
    podem_max_backtracks = max_backtracks

    # Reset per-run statistics (backtrack counts, etc.)
    reset_statistics()

    from src.atpg.util import get_topological_order

    if not scoap_calculated:
        calculate_scoap(circuit, total_gates)
        scoap_calculated = True
        topological_order = get_topological_order(circuit, total_gates)

    # Clear distance maps to force recalculation for each circuit
    gate_distances_back.clear()
    gate_distances_fwd.clear()

    if backtrace_func is not None:
        backtrace_function = backtrace_func
    else:
        backtrace_function = simple_backtrace

    # Calculate distances to primary inputs (backward distances) once
    if gate_distances_back == {}:
        gate_distances_back = calculate_distance_to_primary_inputs(circuit, total_gates)

    # Calculate distances to primary outputs (forward distances) once
    if gate_distances_fwd == {}:
        gate_distances_fwd = calculate_distance_to_primary_outputs(circuit, total_gates)

    # Provide forward distances for D-frontier ordering
    try:
        set_d_frontier_sort(gate_distances_fwd)
    except Exception:
        pass

    # Recurse
    result = podem_recursion(circuit, total_gates, fault)

    if result == SUCCESS:
        if VERBOSE:
            print(f"Fault {fault.gate_id} ({fault.value}) detected")
            print("Pattern:", print_pi(circuit, total_gates))
        return SUCCESS
    else:
        # Provide detailed failure information for debugging
        try:
            gate_type_name = get_gate_type_name(circuit[fault.gate_id].type)
        except Exception:
            gate_type_name = "UNKNOWN"
        fault_label = "D" if fault.value == LogicValue.D else "DB"
        if VERBOSE:
            print(
                f"Fault failed (code={result}): gate_id={fault.gate_id} "
                f"gate_type={gate_type_name} fault={fault_label}"
            )
        return result


# ---------- Recursion & RL backtrace ----------


def podem_recursion(circuit: List[Gate], total_gates: int, fault: Fault) -> int:
    """Core PODEM recursion."""
    global \
        depth, \
        total_recursive_calls, \
        backtrack_count, \
        podem_start_time, \
        podem_timeout, \
        podem_max_backtracks, \
        topological_order, \
        backtrace_function
    depth += 1
    total_recursive_calls += 1

    # Backtrack limit to prevent hanging
    if backtrack_count > podem_max_backtracks:
        return BACKTRACK_LIMIT

    # Wall-clock timeout
    if (time.time() - podem_start_time) > podem_timeout:
        return TIMEOUT

    try:
        if fault_is_at_po(circuit, total_gates):
            return SUCCESS

        objective = get_objective(circuit, fault)
        if objective.gate_id == -1:
            return UNTESTABLE

        # Backtrace to get PI assignment
        pi_assignment = backtrace_function(objective, circuit)
        if pi_assignment.gate_id == -1:
            return UNTESTABLE

        pi_id = pi_assignment.gate_id
        desired_val = pi_assignment.value

        # Try desired value
        circuit[pi_id].val = desired_val
        logic_sim.logic_sim(circuit, total_gates, fault, topo_order=topological_order)

        res = podem_recursion(circuit, total_gates, fault)
        if res == SUCCESS:
            return SUCCESS
        if res != UNTESTABLE:
            return res

        # Backtrack: try flipped value
        circuit[pi_id].val = 1 - desired_val
        logic_sim.logic_sim(circuit, total_gates, fault, topo_order=topological_order)

        res = podem_recursion(circuit, total_gates, fault)
        if res == SUCCESS:
            return SUCCESS
        if res != UNTESTABLE:
            return res

        # Reset PI and return failure
        circuit[pi_id].val = LogicValue.XD
        logic_sim.logic_sim(circuit, total_gates, fault, topo_order=topological_order)
        backtrack_count += 1
        return UNTESTABLE

    finally:
        depth = max(0, depth - 1)


def backtrace_wrapper(objective: Fault, circuit: List[Gate]) -> Fault:
    global backtrace_function, backtrace_count

    backtrace_count += 1

    return backtrace_function(objective, circuit)  # type: ignore


# ---------- Simple backtrace & helpers ----------
def simple_backtrace(objective: Fault, circuit: List[Gate]) -> Fault:
    """
    Simple backtrace function
    Based on backtrace from C implementation
    """
    global backtrace_hops
    num_inv = 0
    current_gate_id = objective.gate_id
    current_gate = circuit[current_gate_id]

    result = Fault(objective.gate_id, objective.value)

    while current_gate.nfi != 0:
        backtrace_hops += 1
        _trace(f"[sb] at gate {current_gate_id} type " f"{get_gate_type_name(current_gate.type)}")

        x_fanins = []

        if current_gate.type in {
            GateType.NOT,
            GateType.NAND,
            GateType.NOR,
            GateType.XNOR,
        }:
            num_inv += 1

        for fanin_id in current_gate.fin:
            if circuit[fanin_id].val == LogicValue.XD:
                x_fanins.append(fanin_id)

        if x_fanins:
            # Use the first X fanin available
            # current_gate_id = x_fanins[0]
            current_gate_id = get_x_fanin(circuit, current_gate_id, gate_distances_back)
            current_gate = circuit[current_gate_id]
            continue
        else:
            # No X fanins available, backtrace failed
            result.gate_id = -1
            return result

    result.gate_id = current_gate_id
    result.value = objective.value if num_inv % 2 == 0 else (1 - objective.value)
    return result


def get_objective(circuit: List[Gate], fault: Fault) -> Fault:
    """
    Get objective for PODEM algorithm
    Based on getObjective from C implementation
    """
    objective = Fault(-1, -1)

    if d_frontier.is_empty() and circuit[fault.gate_id].val in (
        LogicValue.D,
        LogicValue.DB,
    ):
        _trace("[obj] D-frontier empty and fault site is D/DB -> no objective")
        objective.gate_id = -1
        return objective

    if circuit[fault.gate_id].val in (LogicValue.D, LogicValue.DB):
        d_frontier_gate = d_frontier.get_first()
        if d_frontier_gate is not None:
            objective.value = get_non_controlling_value(circuit, d_frontier_gate)
            objective.gate_id = get_x_fanin(circuit, d_frontier_gate, gate_distances_back)
            if objective.gate_id == -1:
                return Fault(-1, -1)

        _trace(
            "[obj] fault site is D/DB -> objective gate "
            f"{objective.gate_id} value {objective.value}"
        )
        return objective

    # C implementation logic:
    # objective.gate_id = fault->gate_id;
    # objective.value = fault->value == D ? 1 : 0;
    objective.gate_id = fault.gate_id
    objective.value = LogicValue.ONE if fault.value == LogicValue.D else LogicValue.ZERO
    _trace(
        "[obj] fault site is not D/DB -> objective gate "
        f"{objective.gate_id} value {objective.value}"
    )
    return objective


def get_non_controlling_value(circuit, gate_id: int | None) -> int:
    if gate_id is None:
        return 0
    t = circuit[gate_id].type
    if t in {GateType.AND, GateType.NAND}:
        return 1
    elif t in {GateType.OR, GateType.NOR}:
        return 0
    elif t in (GateType.XOR, GateType.XNOR):
        # For XOR/XNOR gates, return 0 as the non-controlling value
        # This allows PODEM to properly backtrace through XOR gates
        return 0
    else:
        return 0


def get_all_faults(circuit: List[Gate], total_gates: int) -> List[Fault]:
    """
    Generate Stuck-at-0 and Stuck-at-1 faults for every gate in the circuit.
    """
    faults = []
    # Skip index 0 (if used as placeholder)
    for i in range(1, total_gates + 1):
        if circuit[i] and circuit[i].type != 0:
            faults.append(Fault(i, LogicValue.D))
            faults.append(Fault(i, LogicValue.DB))
    return faults


# -------------------- Minimal CLI runner --------------------
if __name__ == "__main__":
    # Very small runner: python -m src.atpg.podem <path/to/file.bench>
    bench_path = sys.argv[1] if len(sys.argv) > 1 else "data/bench/c432.bench"
    print(f"[PODEM] Running simple mode on: {bench_path}")
    circuit, total_gates = parse_bench_file(bench_path)
    faults = get_all_faults(circuit, total_gates)

    backtrace_function = simple_backtrace

    succ = 0
    fail = 0

    start_time = time.time() * 1000  # milliseconds
    print(f"Starting ATPG... time: {start_time:.2f} milliseconds")

    for k in range(10000):
        for i, fault in enumerate(faults, 1):
            print(
                f"\"\n[PODEM] Processing fault #{i}: gate_id={fault.gate_id} "
                f"value={'D' if fault.value == LogicValue.D else 'DB'}"
            )
            detected = podem(circuit, fault, total_gates)
            succ += 1 if detected else 0
            fail += 0 if detected else 1
            if "tqdm" not in globals() and i % 50 == 0:
                print(f"  processed {i}/{len(faults)} -> succ={succ} fail={fail}")

    end_time = time.time() * 1000  # milliseconds
    print(f"ATPG completed. time: {(end_time - start_time) / 10000:.2f} milliseconds")

    cov = (succ / len(faults) * 100.0) if faults else 0.0
    print("\n=== PODEM simple summary ===")
    print(f"Faults processed : {len(faults)}")
    print(f"Detected/Undetected : {succ}/{fail}")
    print(f"Fault coverage   : {cov:.2f}%")
    print(f"Circuit          : {bench_path}")
