
from typing import List, Dict, Optional
from src.util.struct import Gate, GateType, LogicValue

# Logic Tables
AND_TABLE = [
    [0, 0, 0, 0, 0],
    [0, 1, 2, 3, 4],
    [0, 2, 2, 2, 2],
    [0, 3, 2, 3, 0],
    [0, 4, 2, 0, 4],
]

OR_TABLE = [
    [0, 1, 2, 3, 4],
    [1, 1, 1, 1, 1],
    [2, 1, 2, 2, 2],
    [3, 1, 2, 3, 1],
    [4, 1, 2, 1, 4],
]

NOT_TABLE = [1, 0, 2, 4, 3]

def compute_gate_value(circuit: List[Gate], g: Gate) -> int:
    """Compute the logic value of a gate based on its inputs."""
    if g.type in (GateType.FROM, GateType.BUFF):
        return circuit[g.fin[0]].val
    elif g.type == GateType.NOT:
        return NOT_TABLE[circuit[g.fin[0]].val]
    elif g.type == GateType.AND:
        res = 1
        for fin in g.fin:
            res = AND_TABLE[res][circuit[fin].val]
        return res
    elif g.type == GateType.NAND:
        res = 1
        for fin in g.fin:
            res = AND_TABLE[res][circuit[fin].val]
        return NOT_TABLE[res]
    elif g.type == GateType.OR:
        res = 0
        for fin in g.fin:
            res = OR_TABLE[res][circuit[fin].val]
        return res
    elif g.type == GateType.NOR:
        res = 0
        for fin in g.fin:
            res = OR_TABLE[res][circuit[fin].val]
        return NOT_TABLE[res]
    elif g.type == GateType.XOR:
        # XOR(a, b) = a'b + ab'
        a = circuit[g.fin[0]].val
        b = circuit[g.fin[1]].val
        # Simplified XOR for 5-valued logic
        # 0^0=0, 0^1=1, 1^0=1, 1^1=0
        # For D/DB/X, it gets complex, but this table-based approach works:
        return XOR_TABLE[a][b]
    elif g.type == GateType.XNOR:
        a = circuit[g.fin[0]].val
        b = circuit[g.fin[1]].val
        return NOT_TABLE[XOR_TABLE[a][b]]
    return LogicValue.XD

XOR_TABLE = [
    [0, 1, 2, 3, 4],
    [1, 0, 2, 4, 3],
    [2, 2, 2, 2, 2],
    [3, 4, 2, 0, 1],
    [4, 3, 2, 1, 0],
]

class DFrontier:
    def __init__(self):
        self.gates = []
    def add(self, gate_id):
        if gate_id not in self.gates:
            self.gates.append(gate_id)
    def remove(self, gate_id):
        if gate_id in self.gates:
            self.gates.remove(gate_id)
    def is_empty(self):
        return len(self.gates) == 0
    def get_first(self):
        return self.gates[0] if self.gates else None
    def clear(self):
        self.gates = []
    def sort(self, key_func):
        self.gates.sort(key=key_func)

d_frontier = DFrontier()
_dist_map = {}

def set_d_frontier_sort(distance_map):
    global _dist_map
    _dist_map = distance_map

def logic_sim(circuit: List[Gate], total_gates: int, fault=None, topo_order=None) -> None:
    """Logic simulation. Uses topo_order for single-pass if provided."""
    if topo_order:
        for i in topo_order:
            g = circuit[i]
            if g.type == GateType.INPT:
                if fault and i == fault.gate_id:
                    if fault.value == LogicValue.D and g.val == LogicValue.ONE:
                        g.val = LogicValue.D
                    elif fault.value == LogicValue.DB and g.val == LogicValue.ZERO:
                        g.val = LogicValue.DB
                continue
            
            new_val = compute_gate_value(circuit, g)
            
            if fault and i == fault.gate_id:
                if fault.value == LogicValue.D and new_val == LogicValue.ONE:
                    new_val = LogicValue.D
                elif fault.value == LogicValue.DB and new_val == LogicValue.ZERO:
                    new_val = LogicValue.DB
            g.val = new_val
    else:
        changed = True
        passes = 0
        while changed and passes < 100:
            changed = False
            passes += 1
            for i in range(1, total_gates + 1):
                g = circuit[i]
                old_val = g.val
                if g.type == GateType.INPT:
                    if fault and i == fault.gate_id:
                        if fault.value == LogicValue.D and g.val == LogicValue.ONE:
                            g.val = LogicValue.D
                        elif fault.value == LogicValue.DB and g.val == LogicValue.ZERO:
                            g.val = LogicValue.DB
                    if g.val != old_val: changed = True
                    continue
                
                new_val = compute_gate_value(circuit, g)
                if fault and i == fault.gate_id:
                    if fault.value == LogicValue.D and new_val == LogicValue.ONE:
                        new_val = LogicValue.D
                    elif fault.value == LogicValue.DB and new_val == LogicValue.ZERO:
                        new_val = LogicValue.DB
                
                if new_val != old_val:
                    g.val = new_val
                    changed = True
    
    # Update D-frontier
    d_frontier.clear()
    for i in range(1, total_gates + 1):
        if circuit[i].val == LogicValue.XD:
            for fin in circuit[i].fin:
                if circuit[fin].val in (LogicValue.D, LogicValue.DB):
                    d_frontier.add(i)
                    break
    if _dist_map:
        d_frontier.sort(key_func=lambda gid: _dist_map.get(gid, 999999))

def logic_sim_and_impl(circuit: List[Gate], total_gates: int, fault, assignment) -> None:
    if assignment and assignment.gate_id != -1:
        circuit[assignment.gate_id].val = assignment.value
    logic_sim(circuit, total_gates, fault)

def reset_gates(circuit: List[Gate], total_gates: int) -> None:
    for i in range(total_gates + 1):
        if circuit[i]:
            circuit[i].val = LogicValue.XD
    d_frontier.clear()

def fault_is_at_po(circuit: List[Gate], total_gates: int) -> bool:
    """Check if any primary output has a fault value (D or DB)."""
    for i in range(1, total_gates + 1):
        if circuit[i].type != 0 and circuit[i].nfo == 0:
            if circuit[i].val in (LogicValue.D, LogicValue.DB):
                return True
    return False

def print_pi(circuit: List[Gate], total_gates: int) -> str:
    """Returns string represention of PI values."""
    res = []
    for i in range(1, total_gates + 1):
        if circuit[i].type == GateType.INPT:
            res.append(str(int(circuit[i].val)))
    return "".join(res)
