"""
PODEM (Path-Oriented Decision Making) Algorithm Implementation
MODIFIED to support MLP PPO (stage1), GCN PPO (stage2), and GCN PPO + RND (stage3).
"""

import sys
import time
from typing import List, Tuple, Optional, Callable  # ...updated (added Optional, Any, Callable)
from collections import defaultdict

from src.util.struct import (
    Gate,
    Fault,
    LogicValue,
    GateType,
)
from src.atpg.logic_sim_three import (
    logic_sim_and_impl,
    fault_is_at_po,
    print_pi,
    d_frontier,
    set_d_frontier_sort,
)
import src.atpg.logic_sim_three as logic_sim
from src.atpg.scoap import calculate_scoap
from src.atpg.util import (
    get_x_fanin,
    calculate_distance_to_primary_inputs,
    calculate_distance_to_primary_outputs,
)

# Set maximum recursion depth
sys.setrecursionlimit(500000)

# PODEM constants
FAILURE = 0
SUCCESS = 1
TIMEOUT = 2

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

def podem(
    circuit: List[Gate],
    fault: Fault,
    total_gates: int,
    backtrace_func: Optional[Callable] = None,
    timeout: float = float("inf")
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
        topological_order
        
    podem_start_time = time.time()
    podem_timeout = timeout

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
        return True
    else:
        # Provide detailed failure information for debugging
        try:
            gate_type_name = get_gate_type_name(circuit[fault.gate_id].type)
        except Exception:
            gate_type_name = "UNKNOWN"
        fault_label = "D" if fault.value == LogicValue.D else "DB"
        if VERBOSE:
            print(
                f"Fault untestable: gate_id={fault.gate_id} gate_type={gate_type_name} fault={fault_label}"
            )
        return False


# ---------- Recursion & RL backtrace ----------


def podem_recursion(circuit: List[Gate], total_gates: int, fault: Fault) -> int:
    """Core PODEM recursion."""
    global depth, total_recursive_calls, backtrack_count, podem_start_time, podem_timeout, topological_order, backtrace_function
    depth += 1
    total_recursive_calls += 1
    
    # Backtrack limit to prevent hanging
    if backtrack_count > 1000:
        return FAILURE
        
    # Wall-clock timeout
    if (time.time() - podem_start_time) > podem_timeout:
        return FAILURE

    try:
        if fault_is_at_po(circuit, total_gates):
            return SUCCESS

        objective = get_objective(circuit, fault)
        if objective.gate_id == -1:
            return FAILURE

        # Backtrace to get PI assignment
        pi_assignment = backtrace_function(objective, circuit)
        if pi_assignment.gate_id == -1:
            return FAILURE

        pi_id = pi_assignment.gate_id
        desired_val = pi_assignment.value
        
        # Try desired value
        circuit[pi_id].val = desired_val
        logic_sim.logic_sim(circuit, total_gates, fault, topo_order=topological_order)
        if podem_recursion(circuit, total_gates, fault) == SUCCESS:
            return SUCCESS
            
        # Backtrack: try flipped value
        circuit[pi_id].val = 1 - desired_val
        logic_sim.logic_sim(circuit, total_gates, fault, topo_order=topological_order)
        if podem_recursion(circuit, total_gates, fault) == SUCCESS:
            return SUCCESS
            
        # Reset PI and return failure
        circuit[pi_id].val = LogicValue.XD
        logic_sim.logic_sim(circuit, total_gates, fault, topo_order=topological_order)
        backtrack_count += 1
        return FAILURE

    finally:
        depth = max(0, depth - 1)


def backtrace_wrapper(objective: Fault, circuit: List[Gate]) -> Fault:
    global backtrace_function, backtrace_count

    backtrace_count += 1

    return backtrace_function(objective, circuit) # type: ignore


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
        _trace(
            f"[sb] at gate {current_gate_id} type {get_gate_type_name(current_gate.type)}"
        )

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
            objective.gate_id = get_x_fanin(
                circuit, d_frontier_gate, gate_distances_back
            )
            if objective.gate_id == -1:
                return Fault(-1, -1)

        _trace(
            f"[obj] fault site is D/DB -> objective gate {objective.gate_id} value {objective.value}"
        )
        return objective

    # C implementation logic:
    # objective.gate_id = fault->gate_id;
    # objective.value = fault->value == D ? 1 : 0;
    objective.gate_id = fault.gate_id
    objective.value = LogicValue.ONE if fault.value == LogicValue.D else LogicValue.ZERO
    _trace(
        f"[obj] fault site is not D/DB -> objective gate {objective.gate_id} value {objective.value}"
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


# ---------- Parser & utility (unchanged) ----------
def parse_bench_file(filename: str) -> Tuple[List[Gate], int]:
    circuit: List[Gate] = []
    max_node_id = 0
    inputs, outputs, gates = [], [], []
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("INPUT("):
                node_id = int(line[6:-1])
                inputs.append(node_id)
                max_node_id = max(max_node_id, node_id)
            elif line.startswith("OUTPUT("):
                node_id = int(line[7:-1])
                outputs.append(node_id)
                max_node_id = max(max_node_id, node_id)
            elif "=" in line:
                parts = line.split("=")
                node_id = int(parts[0].strip())
                gate_def = parts[1].strip()
                gate_type_str = gate_def.split("(")[0].strip()
                inputs_str = gate_def.split("(")[1].split(")")[0]
                input_ids = [int(x.strip()) for x in inputs_str.split(",")]
                gates.append((node_id, gate_type_str, input_ids))
                max_node_id = max(max_node_id, node_id)
    circuit = [Gate() for _ in range(max_node_id + 1)]
    for node_id in inputs:
        circuit[node_id] = Gate(str(node_id), GateType.INPT, 0, 0, 0, LogicValue.XD)
    for node_id, gate_type_str, input_ids in gates:
        gate_type = get_gate_type(gate_type_str)
        circuit[node_id] = Gate(
            str(node_id), gate_type, len(input_ids), 0, 0, LogicValue.XD
        )
        circuit[node_id].fin = input_ids
        for input_id in input_ids:
            if circuit[input_id].fot is None:
                circuit[input_id].fot = []
            circuit[input_id].fot.append(node_id)
    # Set fanout counts
    for i in range(1, max_node_id + 1):
        if circuit[i].fot is None:
            circuit[i].fot = []
        circuit[i].nfo = len(circuit[i].fot)
    # Mark outputs (nodes with no fanouts)
    for node_id in outputs:
        circuit[node_id].nfo = 0
    return circuit, max_node_id


def get_gate_type(gate_type_str: str) -> int:
    gate_type_map = {
        "BUFF": GateType.BUFF,
        "NOT": GateType.NOT,
        "AND": GateType.AND,
        "NAND": GateType.NAND,
        "OR": GateType.OR,
        "NOR": GateType.NOR,
        "XOR": GateType.XOR,
        "XNOR": GateType.XNOR,
    }
    return gate_type_map.get(gate_type_str.upper(), GateType.INPT)


def get_all_faults(circuit: List[Gate], total_gates: int) -> List[Fault]:
    """Generate all possible stuck-at faults for the circuit."""
    faults = []
    for gate_id in range(1, total_gates + 1):
        if circuit[gate_id].type != 0:
            faults.append(Fault(gate_id, LogicValue.D))  # SA0
            faults.append(Fault(gate_id, LogicValue.DB))  # SA1
    return faults


def get_gate_type_name(gate_type: int) -> str:
    try:
        return GateType(gate_type).name
    except ValueError:
        return f"UNKNOWN({gate_type})"


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
            print(f"\"\n[PODEM] Processing fault #{i}: gate_id={fault.gate_id} value={'D' if fault.value == LogicValue.D else 'DB'}")
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
