import json
from pathlib import Path

def load_parameters():
    config_path = Path(__file__).resolve().parent.parent / 'config' / 'parameters.json'
    with open(config_path, 'r') as f:
        parameters = json.load(f)
    return parameters
