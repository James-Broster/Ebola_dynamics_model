import pandas as pd
from pathlib import Path

def prepare_data(filename):
    data_path = Path(__file__).resolve().parent / filename
    data = pd.read_csv(data_path)
    return data
