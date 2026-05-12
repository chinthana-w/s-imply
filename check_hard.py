import json
with open("docs/itc99_gate_report_600.json", "r") as f:
    data = json.load(f)
hard_ai = [f for f in data.get("per_fault", []) if f.get("ok") and f.get("classic_backtracks", 0) > 0]
print("Count:", len(hard_ai))

