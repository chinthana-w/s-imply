from typing import List

from src.util.struct import Gate, GateType
from src.util.io import parse_bench_file, write_bench_file

preserve_type = [GateType.INPT, GateType.FROM, GateType.NOT, GateType.AND]

def _create_gate(gate_type: int, gate_id: int, fanins: List[int], fanouts: List[int] = None) -> Gate:
    """Helper function to create a gate with proper fanin/fanout counts."""
    gate = Gate(str(gate_id), gate_type, len(fanins), len(fanouts) if fanouts else 0)
    gate.fin = fanins
    gate.fot = fanouts or []
    return gate

def _add_gate_to_circuit(new_circuit: List[Gate], gate: Gate) -> None:
    """Helper function to add a gate to the circuit."""
    new_circuit.append(gate)

def _convert_or_gate(gate: Gate, new_circuit: List[Gate], next_id: int) -> int:
    """Convert OR gate: OR(a,b) = NOT(AND(NOT(a), NOT(b)))."""
    # Create NOT gates for inputs
    invs = []
    for fanin in gate.fin:
        next_id += 1
        inv = _create_gate(GateType.NOT, next_id, [fanin])
        invs.append(inv)
        _add_gate_to_circuit(new_circuit, inv)

    # Create AND gate for inverted inputs
    next_id += 1
    and_gate = _create_gate(GateType.AND, next_id, [int(i.name) for i in invs])
    _add_gate_to_circuit(new_circuit, and_gate)
    
    # Set fanouts for NOT gates
    for inv in invs:
        inv.fot = [int(and_gate.name)]

    # Create final NOT gate
    next_id += 1
    final_inv = _create_gate(GateType.NOT, next_id, [int(and_gate.name)], [int(x) for x in gate.fot])
    _add_gate_to_circuit(new_circuit, final_inv)
    and_gate.fot = [int(final_inv.name)]
    
    return next_id + 1

def _convert_nor_gate(gate: Gate, new_circuit: List[Gate], next_id: int) -> int:
    """Convert NOR gate: NOR(a,b) = AND(NOT(a), NOT(b))."""
    # Create NOT gates for inputs
    invs = []
    for fanin in gate.fin:
        next_id += 1
        inv = _create_gate(GateType.NOT, next_id, [fanin])
        invs.append(inv)
        _add_gate_to_circuit(new_circuit, inv)

    # Create AND gate for inverted inputs
    next_id += 1
    and_gate = _create_gate(GateType.AND, next_id, [int(i.name) for i in invs], [int(x) for x in gate.fot])
    _add_gate_to_circuit(new_circuit, and_gate)
    
    # Set fanouts for NOT gates
    for inv in invs:
        inv.fot = [int(and_gate.name)]
    
    return next_id

def _convert_xor_gate(gate: Gate, new_circuit: List[Gate], next_id: int) -> int:
    """Convert XOR gate: XOR(a,b) = AND(OR(a,b), NAND(a,b))."""
    # Create NOT gates for inputs
    invs = []
    for fanin in gate.fin:
        next_id += 1
        inv = _create_gate(GateType.NOT, next_id, [fanin])
        invs.append(inv)
        _add_gate_to_circuit(new_circuit, inv)

    # Create AND gate for inverted inputs (NOR part)
    next_id += 1
    nor_gate = _create_gate(GateType.AND, next_id, [int(i.name) for i in invs])
    _add_gate_to_circuit(new_circuit, nor_gate)
    
    # Set fanouts for NOT gates
    for inv in invs:
        inv.fot = [int(nor_gate.name)]

    # Create OR(a,b) = NOT(NOR(a,b))
    next_id += 1
    or_gate = _create_gate(GateType.NOT, next_id, [int(nor_gate.name)])
    _add_gate_to_circuit(new_circuit, or_gate)
    nor_gate.fot = [int(or_gate.name)]
    
    # Create NAND(a,b) = NOT(AND(a,b))
    next_id += 1
    and_ab_gate = _create_gate(GateType.AND, next_id, gate.fin)
    _add_gate_to_circuit(new_circuit, and_ab_gate)
    
    next_id += 1
    nand_gate = _create_gate(GateType.NOT, next_id, [int(and_ab_gate.name)])
    _add_gate_to_circuit(new_circuit, nand_gate)
    and_ab_gate.fot = [int(nand_gate.name)]
    
    # Create final XOR = AND(OR, NAND)
    next_id += 1
    xor_gate = _create_gate(GateType.AND, next_id, [int(or_gate.name), int(nand_gate.name)], [int(x) for x in gate.fot])
    _add_gate_to_circuit(new_circuit, xor_gate)
    
    # Set fanouts
    or_gate.fot = [int(xor_gate.name)]
    nand_gate.fot = [int(xor_gate.name)]
    
    return next_id

def _convert_xnor_gate(gate: Gate, new_circuit: List[Gate], next_id: int) -> int:
    """Convert XNOR gate: XNOR(a,b) = NOT(XOR(a,b))."""
    # Use XOR conversion logic
    next_id = _convert_xor_gate(gate, new_circuit, next_id)
    
    # Find the XOR gate (last gate added) and add final NOT
    xor_gate = new_circuit[-1]  # The XOR gate is the last gate added
    
    # Create final NOT for XNOR
    next_id += 1
    xnor_gate = _create_gate(GateType.NOT, next_id, [int(xor_gate.name)], [int(x) for x in gate.fot])
    _add_gate_to_circuit(new_circuit, xnor_gate)
    xor_gate.fot = [int(xnor_gate.name)]
    
    return next_id

def convert_gate(gate: Gate, new_circuit: List[Gate], next_id: int) -> int:
    """Convert a gate to AIG format. Returns the next available ID."""
    # Handle preserved gate types
    if gate.type in preserve_type:
        new_gate = Gate(gate.name, gate.type, gate.nfi, gate.nfo, gate.mark, gate.val)
        new_gate.fin = gate.fin.copy()
        new_gate.fot = gate.fot.copy()
        new_circuit.append(new_gate)
        return next_id
    
    # Handle gate-specific conversions
    if gate.type == GateType.OR:
        return _convert_or_gate(gate, new_circuit, next_id)
    elif gate.type == GateType.NOR:
        return _convert_nor_gate(gate, new_circuit, next_id)
    elif gate.type == GateType.XOR:
        return _convert_xor_gate(gate, new_circuit, next_id)
    elif gate.type == GateType.XNOR:
        return _convert_xnor_gate(gate, new_circuit, next_id)
    
    # If we reach here, the gate type is not supported
    raise ValueError(f"Unsupported gate type: {gate.type}")

def _get_output_gate_id(gate_type: int, start_id: int, fanin_count: int) -> int:
    """Calculate the output gate ID for converted gates."""
    if gate_type == GateType.OR:
        # OR: 2 NOT + 1 AND + 1 NOT = 4 gates, output is last NOT
        return start_id + 3
    elif gate_type == GateType.NOR:
        # NOR: 2 NOT + 1 AND = 3 gates, output is AND (at start_id + 3)
        return start_id + 3
    elif gate_type == GateType.XOR:
        # XOR: 2 NOT + 1 AND + 1 NOT + 1 AND + 1 NOT + 1 AND = 7 gates, output is last AND
        return start_id + 6
    elif gate_type == GateType.XNOR:
        # XNOR: XOR gates + 1 NOT = 8 gates, output is last NOT
        return start_id + 7
    else:
        return start_id

def _update_gate_connections(gates: List[Gate], idx_map: dict) -> None:
    """Update fanin and fanout connections using the mapping."""
    for gate in gates:
        if gate.fin:
            gate.fin = [int(idx_map.get(str(fanin), fanin)) for fanin in gate.fin]
        if gate.fot:
            gate.fot = [int(idx_map.get(str(fanout), fanout)) for fanout in gate.fot]

def _update_fanout_counts(gates: List[Gate]) -> None:
    """Update fanout counts for all gates."""
    for gate in gates:
        gate.nfo = len(gate.fot) if gate.fot else 0

def _topological_sort(gates: List[Gate]) -> List[Gate]:
    """Sort gates in topological order."""
    gate_map = {g.name: g for g in gates}
    visited = set()
    result = []
    
    def visit(gate_name):
        if gate_name in visited:
            return
        visited.add(gate_name)
        if gate_name in gate_map:
            gate = gate_map[gate_name]
            # Visit all fanins first
            for fanin in gate.fin:
                visit(str(fanin))
            result.append(gate)
    
    # Visit all gates
    for gate in gates:
        visit(gate.name)
    
    return result

def _renumber_gates(gates: List[Gate]) -> dict:
    """Renumber gates to maintain topological ordering. Returns renumbering map."""
    input_gates = [g for g in gates if g.type == GateType.INPT]
    other_gates = [g for g in gates if g.type != GateType.INPT]
    
    # Sort other gates topologically
    other_gates = _topological_sort(other_gates)
    
    # Create renumbering map
    renumber_map = {}
    for gate in input_gates:
        renumber_map[gate.name] = gate.name  # Keep input IDs unchanged
    
    max_input_id = max(int(g.name) for g in input_gates) if input_gates else 0
    next_new_id = max_input_id + 1
    
    for gate in other_gates:
        old_id = gate.name
        new_id = str(next_new_id)
        renumber_map[old_id] = new_id
        gate.name = new_id
        next_new_id += 1
    
    return renumber_map

def bench_to_aig(bench_path: str) -> tuple[List[Gate], dict]:
    """Convert a benchmark file to AIG format."""
    circuit, max_node_id = parse_bench_file(bench_path)
    idx_map = {}
    new_circuit = []
    next_id = max_node_id + 1

    # Convert gates and build mapping
    for node in circuit:
        if node.type > 0:
            original_name = node.name
            
            if node.type in preserve_type:
                # Preserved gates keep their original name
                next_id = convert_gate(node, new_circuit, next_id)
                idx_map[original_name] = original_name
            else:
                # Converted gates need mapping to output gate
                start_id = next_id
                next_id = convert_gate(node, new_circuit, next_id)
                output_gate_id = _get_output_gate_id(node.type, start_id, len(node.fin))
                idx_map[original_name] = str(output_gate_id)
    
    # Update connections using the mapping
    _update_gate_connections(new_circuit, idx_map)
    _update_fanout_counts(new_circuit)
    
    # Renumber gates for topological ordering
    renumber_map = _renumber_gates(new_circuit)
    _update_gate_connections(new_circuit, renumber_map)
    
    # Update final mapping
    final_idx_map = {}
    for original_id, intermediate_id in idx_map.items():
        final_idx_map[original_id] = renumber_map.get(intermediate_id, intermediate_id)
    
    return new_circuit, final_idx_map

if __name__ == "__main__":
    new_circuit, idx_map = bench_to_aig("data/bench/arbitrary/composite_and.bench")
    write_bench_file(new_circuit, "data/bench/arbitrary/composite_and_aig.bench")
    print(idx_map)