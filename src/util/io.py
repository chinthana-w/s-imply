from typing import List, Tuple
from src.util.struct import Gate, GateType, LogicValue

__all__ = ["parse_bench_file"]


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

def get_gate_type_str(gate_type: int) -> str:
    gate_type_map = {
        GateType.BUFF: "BUFF",
        GateType.NOT: "NOT",
        GateType.AND: "AND",
        GateType.NAND: "NAND",
        GateType.OR: "OR",
        GateType.NOR: "NOR",
        GateType.XOR: "XOR",
        GateType.XNOR: "XNOR",
    }
    return gate_type_map.get(gate_type, "INPT") # type: ignore

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


def write_bench_file(circuit: List[Gate], filename: str) -> None:
    with open(filename, "w") as f:
        for node in circuit:
            if node.type == GateType.INPT:
                f.write(f"INPUT({node.name})\n")
            elif node.nfo == 0:
                # Write both gate definition and OUTPUT for output gates
                f.write(f"{node.name} = {get_gate_type_str(node.type)}({','.join([str(x) for x in node.fin])})\n")
                f.write(f"OUTPUT({node.name})\n")
            else:
                f.write(
                    f"{node.name} = {get_gate_type_str(node.type)}({','.join([str(x) for x in node.fin])})\n"
                )


if __name__ == "__main__":
    circuit, max_node_id = parse_bench_file("data/bench/arbitrary/composite_and.bench")
    print(circuit)
    print(max_node_id)
