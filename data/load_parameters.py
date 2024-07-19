import json
from pathlib import Path

def load_parameters(case_type):
    config_path = Path(__file__).resolve().parent.parent / 'config' / 'parameters.json'
    with open(config_path, 'r') as f:
        parameters = json.load(f)
    initial_parameters = parameters['initial_parameters'][case_type]
    bounds = parameters['bounds'][case_type]
    return initial_parameters, bounds
