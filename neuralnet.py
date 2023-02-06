import torch
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F
from tabular_data import *
from torch.utils.data import Dataset, DataLoader, random_split


class AirbnbNightlyPriceImageDataset(Dataset):
    def __init__(self):
        self.data = pd.read_csv('tabular_data/clean_tabular_data.csv')
        self.features, self.label = load_airbnb(self.data, 'Price_Night')
    
    def __getitem__(self, idx):
        return (torch.tensor(self.features[idx]), self.label[idx])

    def __len__(self):
        return len(self.features)

dataset = AirbnbNightlyPriceImageDataset()

# split dataset into training, validation & test sets
train_set, test_set = random_split(dataset, [int(len(dataset) * 0.7), len(dataset) - int(len(dataset) * 0.7)])
test_set, validation_set = random_split(test_set, [int(len(test_set) * 0.5), len(test_set) - int(len(test_set) * 0.5)])
print(
    f'Training set: {len(train_set)} samples, '
    f'Validation set: {len(validation_set)} samples, '
    f'Test set: {len(test_set)} samples'
)

# create dataloaders for each set
batch_size = 8
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

# define neural network architecture
class NN(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        # define layers
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(11, 18),
            torch.nn.ReLU(),
            torch.nn.Linear(18, 1)
        )

    def forward(self, X):
        # returns prediction by passing X through all layers
        return self.layers(X)

model = NN()
# input_tensor = torch.tensor(dataset.features, dtype=torch.float32)
# model(input_tensor)

def train(model, epochs=10):

    optimiser = torch.optim.Adam(model.parameters(), lr= 0.001)

    for epoch in range(epochs):
        for batch in train_loader:
            features, labels = batch
            features = features.type(torch.float32)
            labels = torch.unsqueeze(labels, 1)
            prediction = model(features)
            loss = F.mse_loss(prediction, labels.float())
            loss.backward()
            print(loss.item())
            # optimisation step
            optimiser.step() 
            optimiser.zero_grad()

train(model)
