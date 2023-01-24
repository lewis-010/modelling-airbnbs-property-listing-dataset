import pandas as pd
from tabular_data import *


# preprocessing steps
dataset = pd.read_csv('tabular_data/clean_tabular_data.csv')
features, label = load_airbnb(dataset, 'Category')