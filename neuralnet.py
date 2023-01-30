import torch
import pandas as pd
from tabular_data import *
from torch.utils.data import Dataset, DataLoader, random_split


class AirbnbNightlyPriceImageDataset(Dataset):
    def __init__(self) -> None:
        self.data = pd.read_csv('tabular_data/clean_tabular_data.csv')
        self.features, self.label = load_airbnb(self.data, 'Price_Night')

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        features = self.features[idx,:]
        features = torch.tensor(features, dtype=torch.float32)
        label = self.label[idx].item()
        return (features, label)

dataset = AirbnbNightlyPriceImageDataset()

# split dataset into training, validation & test sets
train_set, test_set = random_split(dataset, [int(len(dataset) * 0.7), len(dataset) - int(len(dataset) * 0.7)])
test_set, validation_set = random_split(test_set, [int(len(test_set) * 0.5), len(test_set) - int(len(test_set) * 0.5)])
print(
    f'Number of samples in training set: {len(train_set)}, '
    f'Number of samples in validation set: {len(validation_set)}, '
    f'Number of samples in test set: {len(test_set)}'
)

# create dataloaders for each set
train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
val_loader = DataLoader(validation_set, batch_size=8, shuffle=True)
test_loader = DataLoader(test_set, batch_size=8, shuffle=True)

example = next(iter(train_loader))
print(example)


class NeuralNet:
    def __init__(self) -> None:
        pass