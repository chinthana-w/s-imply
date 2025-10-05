"""
Logic Simulation Module for PODEM Algorithm
Based on the C implementation from legacy/user.c
"""
from typing import List

from src.util.struct import Gate, GateType, LogicValue

# Lookup tables for gates (based on C implementation)
AND_GATE = [
    [0, 0, 0],
    [0, 1, 2],
    [0, 2, 2],
]

OR_GATE = [
    [0, 1, 2],
    [1, 1, 1],
    [2, 1, 2],
]

XOR_GATE = [
    [0, 1, 2],
    [1, 0, 2],
    [2, 2, 2],
]

NOT_GATE = [1, 0, 2]

def logic_sim(circuit: List[Gate], total_gates: int) -> None:
    """
    Logic simulation with fault injection and implication
    Based on LogicSimAndImpl from C implementation
    """
    # Iterate deterministically from 0..total_gates inclusive.
    # Use a for-loop to guarantee progress even when skipping nodes.
    for node_index in range(0, total_gates + 1):
        current_node = circuit[node_index]

        # Skip if node is not active
        if current_node.type == 0:
            continue
            
        # Logic simulation based on gate type
        if current_node.type == GateType.INPT:
            pass  # Primary input, value already set
        elif current_node.type == GateType.FROM:
            current_node.val = circuit[current_node.fin[0]].val
        elif current_node.type == GateType.BUFF:
            current_node.val = circuit[current_node.fin[0]].val
        elif current_node.type == GateType.NOT:
            current_node.val = NOT_GATE[circuit[current_node.fin[0]].val]

        elif current_node.type == GateType.AND:
            node_result = 1
            for fanin_id in current_node.fin:
                if node_result == 0:
                    break
                node_result = AND_GATE[node_result][circuit[fanin_id].val]

            current_node.val = node_result

        elif current_node.type == GateType.NAND:
            node_result = 1
            for fanin_id in current_node.fin:
                if node_result == 0:
                    break
                node_result = AND_GATE[node_result][circuit[fanin_id].val]

            current_node.val = NOT_GATE[node_result]

        elif current_node.type == GateType.OR:
            node_result = 0
            for fanin_id in current_node.fin:
                if node_result == 1:
                    break
                node_result = OR_GATE[node_result][circuit[fanin_id].val]

            current_node.val = node_result

        elif current_node.type == GateType.NOR:
            node_result = 0
            for fanin_id in current_node.fin:
                if node_result == 1:
                    break
                node_result = OR_GATE[node_result][circuit[fanin_id].val]

            current_node.val = NOT_GATE[node_result]

        elif current_node.type == GateType.XOR:
            node_result = circuit[current_node.fin[0]].val
            for fanin_id in current_node.fin[1:]:
                if node_result == LogicValue.XD:
                    break
                node_result = XOR_GATE[node_result][circuit[fanin_id].val]

            current_node.val = node_result

        elif current_node.type == GateType.XNOR:
            node_result = circuit[current_node.fin[0]].val
            for fanin_id in current_node.fin[1:]:
                if node_result == LogicValue.XD:
                    break
                node_result = XOR_GATE[node_result][circuit[fanin_id].val]

            current_node.val = NOT_GATE[node_result]

def reset_gates(circuit: List[Gate], total_gates: int):
    """
    Reset all gates to unknown state
    Based on reset_gates from C implementation
    """
    for node_index in range(total_gates + 1):
        circuit[node_index].val = LogicValue.XD

def print_pi(circuit: List[Gate], total_gates: int) -> str:
    """
    Print primary input values
    Based on printPI from C implementation
    """
    result = ""
    for i in range(1, total_gates + 1):
        if circuit[i].type != 0 and circuit[i].nfi == 0:
            if circuit[i].val == LogicValue.XD:
                result += "X"
            elif circuit[i].val == LogicValue.D:
                result += "1"  # D represents good=1, faulty=0
            elif circuit[i].val == LogicValue.DB:
                result += "0"  # D-bar represents good=0, faulty=1
            elif circuit[i].val == 0:
                result += "0"
            elif circuit[i].val == 1:
                result += "1"
            else:
                result += "X"  # Default to X for unexpected values
    return result

def print_po(circuit: List[Gate], total_gates: int) -> str:
    """
    Print primary output values
    Based on printPO from C implementation
    """
    result = ""
    for i in range(1, total_gates + 1):
        if circuit[i].type != 0 and circuit[i].nfo == 0:
            result += str(circuit[i].val)
    return result
