import json
from pathlib import Path
from .types import to_dict
class JsonlRecorder:
    def __init__(self, output_path:str):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        self.fp=open(output_path,"a",encoding="utf-8")
    def write(self, record:dict): self.fp.write(json.dumps(record)+"\n")
    def close(self): self.fp.close()
