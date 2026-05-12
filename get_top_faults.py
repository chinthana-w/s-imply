import json

with open("docs/itc99_gate_report_600.json", "r") as f:
    data = json.load(f)

faults = data.get("per_fault", [])
# Filter only faults where AI succeeded
succeeded_faults = [f for f in faults if f.get("ok")]
sorted_faults = sorted(succeeded_faults, key=lambda x: x.get("classic_backtracks", 0), reverse=True)[:20]

print("| Gate ID | Fault Val | Classic Backtracks | AI Success | Classic Success |")
print("|---|---|---|---|---|")
for f in sorted_faults:
    print(f"| {f.get('gate_id')} | {f.get('fault_val')} | {f.get('classic_backtracks')} | {f.get('ok')} | {f.get('classic_ok')} |")
