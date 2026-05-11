#!/usr/bin/env python
"""
Verification script for bench file parser - checks that nodes are correctly labeled
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.util.io import parse_bench_file
from src.util.struct import GateType


def verify_c17_parsing():
    """Verify that c17.bench is parsed correctly"""

    print("=" * 80)
    print("VERIFYING c17.bench PARSING")
    print("=" * 80)

    circuit, total_gates = parse_bench_file("data/bench/ISCAS85/c17.bench")

    print(f"\nTotal gates in circuit: {len(circuit)}")
    print(f"Max node ID: {total_gates}")

    # Expected structure from c17.bench:
    # INPUT(1), INPUT(2), INPUT(3), INPUT(6), INPUT(7)
    # OUTPUT(22), OUTPUT(23)
    # Gates: 10, 11, 16, 19, 22, 23 (all NAND)

    expected_inputs = [1, 2, 3, 6, 7]
    expected_gates = {
        10: ("NAND", [1, 3]),
        11: ("NAND", [3, 6]),
        16: ("NAND", [2, 11]),
        19: ("NAND", [11, 7]),
        22: ("NAND", [19, 16]),
        23: ("NAND", [16, 10]),
    }

    gate_type_names = {
        0: "UNINITIALIZED",
        GateType.INPT: "INPUT",
        GateType.FROM: "BRANCH",
        GateType.BUFF: "BUFF",
        GateType.NOT: "NOT",
        GateType.AND: "AND",
        GateType.NAND: "NAND",
        GateType.OR: "OR",
        GateType.NOR: "NOR",
        GateType.XOR: "XOR",
        GateType.XNOR: "XNOR",
    }

    print("\n" + "=" * 80)
    print("CHECKING INPUT NODES")
    print("=" * 80)

    all_inputs_correct = True
    for node_id in expected_inputs:
        gate = circuit[node_id]
        type_name = gate_type_names.get(gate.type, f"UNKNOWN({gate.type})")
        is_correct = gate.type == GateType.INPT
        status = "✓" if is_correct else "✗"

        print(
            f"{status} Node {node_id}: type={type_name}, "
            f"fanins={gate.fin if hasattr(gate, 'fin') else 'N/A'}, "
            f"fanouts={gate.fot if hasattr(gate, 'fot') else 'N/A'}"
        )

        if not is_correct:
            all_inputs_correct = False
            print(f"   ERROR: Expected type INPUT (1), got {type_name} ({gate.type})")

    print("\n" + "=" * 80)
    print("CHECKING GATE NODES")
    print("=" * 80)

    all_gates_correct = True
    for node_id, (expected_type, expected_fanins) in expected_gates.items():
        gate = circuit[node_id]
        type_name = gate_type_names.get(gate.type, f"UNKNOWN({gate.type})")
        is_correct_type = type_name == expected_type
        is_correct_fanins = gate.fin == expected_fanins if hasattr(gate, "fin") else False
        is_correct = is_correct_type and is_correct_fanins
        status = "✓" if is_correct else "✗"

        print(
            f"{status} Node {node_id}: type={type_name}, "
            f"fanins={gate.fin if hasattr(gate, 'fin') else 'N/A'}, "
            f"fanouts={gate.fot if hasattr(gate, 'fot') else 'N/A'}"
        )

        if not is_correct_type:
            all_gates_correct = False
            print(f"   ERROR: Expected type {expected_type}, got {type_name}")
        if not is_correct_fanins:
            all_gates_correct = False
            print(
                f"   ERROR: Expected fanins {expected_fanins}, "
                f"got {gate.fin if hasattr(gate, 'fin') else 'N/A'}"
            )

    print("\n" + "=" * 80)
    print("CHECKING UNINITIALIZED NODES")
    print("=" * 80)

    # Nodes that should NOT be initialized (gaps in node numbering)
    all_node_ids = set(expected_inputs) | set(expected_gates.keys())
    uninitialized_found = []

    for i in range(len(circuit)):
        if i not in all_node_ids and i > 0:  # Skip index 0
            gate = circuit[i]
            if gate.type != 0:  # Should be uninitialized (type 0)
                type_name = gate_type_names.get(gate.type, f"UNKNOWN({gate.type})")
                print(f"⚠ Node {i}: Unexpectedly initialized as {type_name}")
                uninitialized_found.append(i)

    if not uninitialized_found:
        print("✓ All gap nodes are properly uninitialized")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    if all_inputs_correct and all_gates_correct and not uninitialized_found:
        print("✓ ALL CHECKS PASSED - c17.bench parsed correctly!")
        return True
    else:
        print("✗ SOME CHECKS FAILED:")
        if not all_inputs_correct:
            print("  - Input nodes incorrectly labeled")
        if not all_gates_correct:
            print("  - Gate nodes incorrectly labeled")
        if uninitialized_found:
            print(f"  - {len(uninitialized_found)} gap nodes unexpectedly initialized")
        return False


if __name__ == "__main__":
    success = verify_c17_parsing()
    sys.exit(0 if success else 1)
