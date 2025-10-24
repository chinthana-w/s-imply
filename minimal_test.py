#!/usr/bin/env python
"""Minimal test to debug reconv_justify."""
import sys
from src.util.io import parse_bench_file
from src.util.struct import LogicValue

print("Step 1: Loading circuit", file=sys.stderr)
circuit, _ = parse_bench_file('data/bench/ISCAS85/c17.bench')
print(f"Circuit loaded with {len(circuit)} gates", file=sys.stderr)

print("\nStep 2: Checking circuit structure", file=sys.stderr)
for i in range(min(20, len(circuit))):
    gate = circuit[i]
    if hasattr(gate, 'name'):
        print(f"Gate {i}: name={gate.name}, type={getattr(gate, 'type', '?')}, fin={getattr(gate, 'fin', [])}", file=sys.stderr)

print("\nTest complete!", file=sys.stderr)
