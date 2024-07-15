import pandas as pd
from pathlib import Path

def prepare_data():
    data_path = Path(__file__).resolve().parent / 'viral_load.csv'
    data = pd.read_csv(data_path)
    return data
