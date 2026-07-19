"""Convert symbolic sequential ITC99 BENCH files to numeric full-scan BENCH files."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

PORT_RE = re.compile(r"^(INPUT|OUTPUT)\(([^)]+)\)$", re.IGNORECASE)
GATE_RE = re.compile(r"^([^=]+)=\s*([A-Za-z0-9_]+)\(([^)]*)\)$")


def _unique(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


def normalize(source: Path, destination: Path) -> dict:
    inputs: list[str] = []
    outputs: list[str] = []
    gates: list[tuple[str, str, list[str]]] = []
    comments: list[str] = []

    for raw_line in source.read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#"):
            comments.append(line)
            continue
        port_match = PORT_RE.match(line)
        if port_match:
            target = inputs if port_match.group(1).upper() == "INPUT" else outputs
            target.append(port_match.group(2).strip())
            continue
        gate_match = GATE_RE.match(line)
        if not gate_match:
            raise ValueError(f"unsupported BENCH line in {source}: {line!r}")
        fanins = [item.strip() for item in gate_match.group(3).split(",") if item.strip()]
        gates.append((gate_match.group(1).strip(), gate_match.group(2).upper(), fanins))

    dffs = [(lhs, fanins[0]) for lhs, kind, fanins in gates if kind == "DFF"]
    dff_outputs = {lhs for lhs, _ in dffs}
    scan_inputs = _unique(inputs + [lhs for lhs, _ in dffs])
    scan_outputs = _unique(
        [name for name in outputs if name not in dff_outputs] + [data for _, data in dffs]
    )
    combinational = [gate for gate in gates if gate[1] != "DFF"]

    names: list[str] = []
    names.extend(scan_inputs)
    names.extend(scan_outputs)
    for lhs, _, fanins in combinational:
        names.append(lhs)
        names.extend(fanins)
    mapping = {name: index for index, name in enumerate(_unique(names), start=1)}

    defined = set(scan_inputs) | {lhs for lhs, _, _ in combinational}
    referenced = {fanin for _, _, fanins in combinational for fanin in fanins}
    undefined = sorted(referenced - defined)
    if undefined:
        raise ValueError(f"undefined signals in {source}: {undefined[:20]}")

    lines = [
        f"# Numeric full-scan conversion of {source}",
        f"# Original inputs={len(inputs)} outputs={len(outputs)} dffs={len(dffs)}",
        f"# Scan inputs={len(scan_inputs)} scan outputs={len(scan_outputs)}",
        "",
    ]
    lines.extend(f"INPUT({mapping[name]})" for name in scan_inputs)
    lines.append("")
    lines.extend(f"OUTPUT({mapping[name]})" for name in scan_outputs)
    lines.append("")
    for lhs, kind, fanins in combinational:
        encoded_fanins = ", ".join(str(mapping[name]) for name in fanins)
        lines.append(f"{mapping[lhs]} = {kind}({encoded_fanins})")

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines) + "\n")
    return {
        "source": str(source),
        "destination": str(destination),
        "original_inputs": len(inputs),
        "original_outputs": len(outputs),
        "dffs": len(dffs),
        "scan_inputs": len(scan_inputs),
        "scan_outputs": len(scan_outputs),
        "combinational_gates": len(combinational),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    sources = sorted(Path(args.input_dir).glob("*.bench"))
    if not sources:
        raise RuntimeError(f"no .bench files found in {args.input_dir}")
    for source in sources:
        stats = normalize(source, Path(args.output_dir) / source.name)
        print(
            f"{source.name}: scan_pi={stats['scan_inputs']} "
            f"scan_po={stats['scan_outputs']} gates={stats['combinational_gates']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
