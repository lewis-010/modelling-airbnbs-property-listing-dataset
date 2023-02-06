import os
import itertools
import torch
import json
import time
import yaml
import numpy as np
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from tabular_data import *
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter


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


def get_nn_config():
    with open("nn_config.yaml", "r") as stream:
        hyper_dict = yaml.safe_load(stream)
        print(hyper_dict)

    return hyper_dict

    
# define neural network architecture
class NN(nn.Module):

    def __init__(self, config):
        super().__init__()
        # define layers
        width = config['hidden_layer_width']
        depth = config['depth']     

        self.layers = torch.nn.Sequential(
            torch.nn.Linear(11, 18),
            torch.nn.ReLU(),
            torch.nn.Linear(18, 1)
        )

    def forward(self, X):
        # returns prediction by passing X through all layers
        return self.layers(X)


def train(model, hyper_dict, epochs=10):

    optimiser_class = hyper_dict['optimiser']
    optimiser_instance = getattr(torch.optim, optimiser_class)
    optimiser = optimiser_instance(model.parameters(), lr=hyper_dict['learning_rate'])

    writer = SummaryWriter()

    batch_idx = 0

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
            # add loss metric to tensorboard
            writer.add_scalar('training_loss', loss.item(), batch_idx)
            batch_idx += 1


def evaluate_model(model, training_duration, epochs):

    scaler = StandardScaler()
    metrics = {'training_duration': training_duration}
    number_of_preds = epochs * len(train_set)
    inference_latency = training_duration / number_of_preds
    metrics['inference_latency'] = inference_latency

    X_train = torch.stack([tuple[0] for tuple in train_set]).type(torch.float32)
    # scaler.fit(X_train)
    # X_train_scaled = scaler.transform(X_train)
    y_train = torch.stack([torch.tensor(tuple[1]) for tuple in train_set])
    y_train = torch.unsqueeze(y_train, 1)
    y_train = torch.tensor([data[1] for data in train_set])
    y_train_pred = model(X_train)
    train_rmse_loss = torch.sqrt(F.mse_loss(y_train_pred, y_train.float()))
    train_r2_score = 1 - train_rmse_loss / torch.var(y_train.float())

    print('Train_RMSE: ', train_rmse_loss.item())
    print('Train_R2: ', train_r2_score.item())

    X_test = torch.stack([tuple[0] for tuple in test_set]).type(torch.float32)
    # scaler.fit(X_test)
    # X_test_scaled = scaler.transform(X_test)
    y_test = torch.stack([torch.tensor(tuple[1]) for tuple in test_set])
    y_test = torch.unsqueeze(y_test, 1)
    y_test_pred = model(X_test)
    test_rmse_loss = torch.sqrt(F.mse_loss(y_test_pred, y_test.float()))
    test_r2_score = 1 - test_rmse_loss / torch.var(y_test.float())

    print('Test_RMSE: ', test_rmse_loss.item())
    print('Test_R2: ', test_r2_score.item())


    X_validation = torch.stack([tuple[0] for tuple in validation_set]).type(torch.float32)
    # scaler.fit(X_validation)    
    # X_validation_scaled = scaler.transform(X_validation)
    y_validation = torch.stack([torch.tensor(tuple[1]) for tuple in validation_set])
    y_validation = torch.unsqueeze(y_validation, 1)
    y_validation_pred = model(X_validation)
    validation_rmse_loss = torch.sqrt(F.mse_loss(y_validation_pred, y_validation.float()))
    validation_r2_score = 1 - validation_rmse_loss / torch.var(y_validation.float())

    print('validation_RMSE: ', validation_rmse_loss.item())
    print('validation_R2: ', validation_r2_score.item())


    RMSE_loss = [train_rmse_loss, validation_rmse_loss, test_rmse_loss]
    R_squared = [train_r2_score, validation_r2_score, test_r2_score]

    metrics["RMSE_loss"] = [loss.item() for loss in RMSE_loss]
    metrics["R_squared"] = [score.item() for score in R_squared]

    return metrics


def save_model(model, hyper_dict, metrics):

    model_name = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    model_folder = 'neural_networks/regression/' + model_name
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    
    torch.save(model.state_dict(), f'{model_folder}/model.pt')

    with open(f'{model_folder}/hyperparameters.json', 'w') as file:
        json.dump(hyper_dict, file)

    with open(f'{model_folder}/metrics.json', 'w') as file:
        json.dump(metrics, file)


def generate_nn_configs():
    hyper_dict_list = []
    param_grid = {
        'optimiser': ['Adam', 'SDG'],
        'learning_rate': [0.0005, 0.001, 0.015],
        'hidden_layer_width': [12, 16, 20],
        'depth': [3, 6, 9]
    }
    for values in itertools.product(*param_grid.values()):
        hyper_dict_list.append(dict(zip(param_grid, values)))
    
    return hyper_dict_list


def evaluate_all_models(hyper_dict):
    model = NN(hyper_dict)
    start_time = time.time()
    train(model, hyper_dict, epochs=10)
    end_time = time.time()
    training_duration = end_time - start_time
    metrics = evaluate_model(model, training_duration, epochs=10)
    save_model(model, hyper_dict, metrics)
    model_data = [model, hyper_dict, metrics]

    return model_data


def find_best_nn():

    best_val_rmse = np.inf
    best_val_r2 = -np.inf


    hyper_dict_list = generate_nn_configs()
    for hyper_dict in hyper_dict_list:
        model_data = evaluate_all_models(hyper_dict)
        metrics = model_data[2]

        rmse_loss = metrics['RMSE_loss']
        rmse_validation_loss = rmse_loss[2]

        r_squared = metrics['R_squared']
        r_squared_val = r_squared[2]

        if rmse_validation_loss < best_val_rmse and r_squared_val > best_val_r2:
            best_val_rmse = rmse_validation_loss
            best_val_r2 = r_squared_val
            best_model_data = model_data
        
    best_model, best_hyper_dict, best_metrics = best_model_data
    print(best_model)

    save_model(best_model, best_hyper_dict, best_metrics)

if __name__ == '__main__':
    find_best_nn()