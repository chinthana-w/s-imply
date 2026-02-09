#!/usr/bin/env python
"""
Debug script for path picker - visualizes reconvergent path pairs
"""
import sys
import os
from datetime import datetime
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.util.io import parse_bench_file
from src.atpg.recursive_reconv_solver import HierarchicalReconvSolver
from src.atpg.ai_podem import ModelPairPredictor
from src.util.struct import LogicValue, Fault, GateType

def print_gate_info(circuit, gate_id):
    """Print detailed gate information"""
    if gate_id >= len(circuit):
        return f"Gate {gate_id}: OUT OF BOUNDS"
    
    gate = circuit[gate_id]
    
    # Use actual GateType enum
    gate_type_names = {
        0: 'UNINITIALIZED',
        GateType.INPT: 'INPUT',
        GateType.FROM: 'BRANCH',
        GateType.BUFF: 'BUFF',
        GateType.NOT: 'NOT',
        GateType.AND: 'AND',
        GateType.NAND: 'NAND',
        GateType.OR: 'OR',
        GateType.NOR: 'NOR',
        GateType.XOR: 'XOR',
        GateType.XNOR: 'XNOR'
    }
    type_name = gate_type_names.get(gate.type, f'UNKNOWN({gate.type})')
    
    fanins = gate.fin if hasattr(gate, 'fin') and gate.fin is not None else []
    fanouts = gate.fot if hasattr(gate, 'fot') and gate.fot is not None else []
    
    return f"Gate {gate_id}: {type_name} (fanins: {fanins}, fanouts: {fanouts})"

def print_path(circuit, path, indent="  "):
    """Print a path in human-readable format"""
    print(f"{indent}Path length: {len(path)}")
    for i, node_id in enumerate(path):
        arrow = " -> " if i < len(path) - 1 else ""
        print(f"{indent}  [{i}] {print_gate_info(circuit, node_id)}{arrow}")

def debug_path_picker(output_file=None):
    """Debug the path picker for c432 circuit, fault sa0 on node 296"""
    
    # Redirect output to file if specified
    if output_file:
        sys.stdout = open(output_file, 'w')
    
    print("="*80)
    print("DEBUG: Path Picker for c432, Fault sa0 on Node 296")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Parse circuit
    circuit_path = 'data/bench/ISCAS85/c432.bench'
    circuit, total_gates = parse_bench_file(circuit_path)
    
    print(f"\nCircuit: c432")
    print(f"Total gates: {len(circuit)}")
    print(f"Total gates (including PIs): {total_gates}")
    
    # Print circuit structure
    print("\n" + "="*80)
    print("CIRCUIT STRUCTURE")
    print("="*80)
    for i in range(len(circuit)):
        gate = circuit[i]
        # Only print initialized nodes
        if gate.type != 0:
            print(print_gate_info(circuit, i))
    
    # Create fault: sa0 on node 22
    fault = Fault(gate_id=296, value=LogicValue.D)  # D means stuck-at-0
    print(f"\n" + "="*80)
    print(f"FAULT: Gate {fault.gate_id} stuck-at-0")
    print(print_gate_info(circuit, fault.gate_id))
    print("="*80)
    
    # Create predictor and solver
    print("\nInitializing predictor and solver...")
    try:
        predictor = ModelPairPredictor(
            circuit_path, 
            'checkpoints/reconv_rl_model.pt', 
            circuit
        )
        solver = HierarchicalReconvSolver(circuit, predictor)
        print("✓ Predictor and solver initialized")
    except Exception as e:
        print(f"✗ Failed to initialize: {e}")
        return
    
    # Get transitive fanin
    print(f"\n" + "="*80)
    print(f"TRANSITIVE FANIN of Gate {fault.gate_id}")
    print("="*80)
    fanin_set = solver._get_transitive_fanin(fault.gate_id)
    print(f"Fanin nodes: {sorted(fanin_set)}")
    print(f"Total fanin nodes: {len(fanin_set)}")
    
    # Find reconvergent pairs
    print(f"\n" + "="*80)
    print("RECONVERGENT PAIRS")
    print("="*80)
    pairs = solver._find_pairs_in_set(fanin_set)
    print(f"Found {len(pairs)} reconvergent pair(s)")
    
    if not pairs:
        print("\n⚠ No reconvergent pairs found!")
        print("This means the circuit has no reconvergent fanout structure")
        print("in the transitive fanin cone of the fault node.")
        return
    
    # Print each pair in detail
    for idx, pair in enumerate(pairs):
        print(f"\n{'-'*80}")
        print(f"PAIR #{idx + 1}")
        print(f"{'-'*80}")
        
        # Check which keys are available
        stem_key = 'start' if 'start' in pair else 'stem'
        stem_node = pair[stem_key]
        
        print(f"Stem/Start node: {stem_node}")
        print(print_gate_info(circuit, stem_node))
        print(f"\nReconvergence node: {pair['reconv']}")
        print(print_gate_info(circuit, pair['reconv']))
        
        if 'branches' in pair:
            print(f"\nBranch nodes: {pair['branches']}")
        
        print(f"\nPaths ({len(pair['paths'])} total):")
        for path_idx, path in enumerate(pair['paths']):
            print(f"\n  Path {path_idx + 1}:")
            print_path(circuit, path, indent="    ")
    
    # Collect and sort all pairs
    print(f"\n" + "="*80)
    print("SORTED PAIRS (by priority)")
    print("="*80)
    sorted_pairs = solver._collect_and_sort_pairs(fault.gate_id)
    print(f"Total pairs after sorting: {len(sorted_pairs)}")
    
    for idx, pair in enumerate(sorted_pairs[:5]):  # Show top 5
        stem_key = 'start' if 'start' in pair else 'stem'
        print(f"\n{idx + 1}. Stem: {pair[stem_key]}, Reconv: {pair['reconv']}, "
              f"Paths: {len(pair['paths'])}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Debug path picker for ATPG')
    parser.add_argument('--output', '-o', default='debug_path_picker_output.txt',
                        help='Output file path (default: debug_path_picker_output.txt)')
    args = parser.parse_args()
    
    debug_path_picker(output_file=args.output)
    print(f"\nOutput written to: {args.output}", file=sys.stderr)
