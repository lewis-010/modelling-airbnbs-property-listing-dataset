import os
import itertools
import torch
import json
import time
import yaml
import shutil
import numpy as np
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
from datetime import datetime
from tabular_data import *
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter


class AirbnbBedroomDataset(Dataset):
    def __init__(self):
        self.data = pd.read_csv('tabular_data/clean_tabular_data.csv')
        numerical_data = load_airbnb(self.data, 'bedrooms')
        category_column = self.data['Category']
        category_data = category_column.unique()
        encoder = LabelEncoder()
        category_encoded = encoder.fit_transform(category_column)

        numerical_data = numerical_data[0]
        numerical_df = pd.DataFrame(data=numerical_data, columns=['bedrooms_' + str(i) for i in range(numerical_data.shape[1])])
        category_df = pd.DataFrame(data=category_encoded, columns=['Category_Encoded'])
      
        self.features = pd.concat([numerical_df, category_df], axis=1)
        self.label = self.data['bedrooms']

    def __getitem__(self, idx):
        return (torch.tensor(self.features.iloc[idx].values), self.label[idx])

    def __len__(self):
        return len(self.features)

dataset = AirbnbBedroomDataset()

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

    def __init__(self, config):
        super().__init__()
        # get values for width & depth from the config
        width = config['hidden_layer_width']
        depth = config['depth']
        dropout_prob = config.get('dropout_prob', 0) # add optional dropout probability

        # define the layers
        layers = [torch.nn.Linear(12, width), torch.nn.ReLU()]
        for hidden_layer in range(depth - 1):
            layers.extend([torch.nn.Dropout(dropout_prob), torch.nn.Linear(width, width), torch.nn.ReLU()]) # add dropout layer
        layers.extend([torch.nn.Linear(width, 1)])
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, X):
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

    metrics = {'training_duration': training_duration}
    number_of_preds = epochs * len(train_set)
    inference_latency = training_duration / number_of_preds
    metrics['inference_latency'] = inference_latency

    X_train = torch.stack([tuple[0] for tuple in train_set]).type(torch.float32)
    y_train = torch.stack([torch.tensor(tuple[1]) for tuple in train_set])
    y_train = torch.unsqueeze(y_train, 1)
    y_train = torch.tensor([data[1] for data in train_set])
    y_train_pred = model(X_train)
    train_rmse_loss = torch.sqrt(F.mse_loss(y_train_pred, y_train.float()))
    train_r2_score = 1 - train_rmse_loss / torch.var(y_train.float())

    print('Train_RMSE: ', train_rmse_loss.item())
    print('Train_R2: ', train_r2_score.item())

    X_test = torch.stack([tuple[0] for tuple in test_set]).type(torch.float32)
    y_test = torch.stack([torch.tensor(tuple[1]) for tuple in test_set])
    y_test = torch.unsqueeze(y_test, 1)
    y_test_pred = model(X_test)
    test_rmse_loss = torch.sqrt(F.mse_loss(y_test_pred, y_test.float()))
    test_r2_score = 1 - test_rmse_loss / torch.var(y_test.float())

    print('Test_RMSE: ', test_rmse_loss.item())
    print('Test_R2: ', test_r2_score.item())

    X_validation = torch.stack([tuple[0] for tuple in validation_set]).type(torch.float32)
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

    model_name = datetime.now().strftime('%Y-%m-%d_%H.%M.%S')
    model_folder = 'neural_networks/regression_bedrooms/' + model_name
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
        'optimiser': ['Adam', 'AdamW'],
        'learning_rate': [0.0005, 0.001],
        'hidden_layer_width': [5, 10, 15],
        'depth': [2, 3, 4],
        'dropout_prob': [0.2, 0.5, 0.75]
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

    best_model_folder = 'neural_networks/regression_bedrooms/best_model'
    if os.path.exists(best_model_folder):
        shutil.rmtree(best_model_folder)

    os.makedirs(best_model_folder)

    torch.save(best_model.state_dict(), f'{best_model_folder}/model.pt')
    with open(f'{best_model_folder}/hyperparameters.json', 'w') as file:
        json.dump(best_hyper_dict, file)

    with open(f'{best_model_folder}/metrics.json', 'w') as file:
        json.dump(best_metrics, file)

    print(best_model)


if __name__ == '__main__':
    with open('neural_networks/regression_bedrooms/best_model/hyperparameters.json', 'r') as file:
        config = json.load(file)

    model = NN(config)
    model.load_state_dict(torch.load('neural_networks/regression_bedrooms/best_model/model.pt'))
    model.eval()

    # preprocessing steps
    test_dataset = AirbnbBedroomDataset()
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # get predictions from the model
    predictions = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.float()
            outputs = model(inputs)
            predictions.append(outputs)
            
    # convert the predictions list to a numpy array & display as graph
    predictions = torch.cat(predictions, dim=0).numpy()
    plt.plot(predictions)
    plt.title('Neural network')
    plt.xlabel('Sample')
    plt.ylabel('Number of Bedrooms')
    plt.show()