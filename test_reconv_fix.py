"""Test script to verify the reconv_justify fix."""

from src.util.io import parse_bench_file
from src.util.struct import LogicValue
from src.atpg.reconv_podem import (
    pick_reconv_pair,
    build_flattened_path,
    reset_circuit_values,
    reconv_justify,
    extract_justification
)

print("Loading circuit...")
# Load a small circuit
circuit, _ = parse_bench_file('data/bench/ISCAS85/c499.bench')

print("Finding reconvergent structure...")
# Get a reconvergent path
info = pick_reconv_pair(circuit, beam_width=16, max_depth=25)

if info:
    print("Found reconvergent structure:")
    print(f"  Start: {info['start']}")
    print(f"  Reconv: {info['reconv']}")
    print(f"  Branches: {info['branches']}")
    print(f"  Paths: {info['paths']}")
    print()
    
    # Build flattened path
    print("Building flattened path...")
    build_flattened_path(info)
    
    # Test justification for LogicValue.ONE
    print("Testing justification for ONE...")
    reset_circuit_values(circuit, info)
    success_1 = reconv_justify(circuit, info, info['reconv'], LogicValue.ONE)
    j1 = extract_justification(circuit, info)
    
    print(f"Justification for ONE (success={success_1}):")
    for name, val in sorted(j1.items()):
        val_name = "ZERO" if val == LogicValue.ZERO else "ONE" if val == LogicValue.ONE else "XD"
        print(f"  {name}: {val_name} ({val})")
    print()
    
    # Count XD values
    xd_count = sum(1 for v in j1.values() if v == LogicValue.XD)
    print(f"Number of XD (unassigned) values: {xd_count}")
    
    if xd_count > 0:
        print("WARNING: Some gates were not assigned values!")
        xd_gates = [name for name, val in j1.items() if val == LogicValue.XD]
        print(f"Unassigned gates: {xd_gates}")
else:
    print("No reconvergent structure found")
