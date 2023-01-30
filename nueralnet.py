import torch
import pandas as pd
from tabular_data import *


class AirbnbNightlyPriceImageDataset(Dataset):
    def __init__(self):
        self.data = pd.read_csv('tabular_data/clean_tabular_data.csv')
        self.features, self.label = load_airbnb(self.data, 'Price_Night')

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        features = self.features.iloc[idx]
        features = torch.tensor(features)
        label = self.label.iloc[idx]
        return (features, label)


