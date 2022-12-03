import pandas as pd

df = pd.read_csv('tabular_data/AirBnbData.csv')

def remove_rows_with_missing_ratings():
    df