import json
import numpy as np
import os
from joblib import dump
from tabular_data import *
from sklearn.linear_model import SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split

param_grid_sgd = {
    'loss': ['squared_error', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'],
    'penalty': ['l2', 'l1', 'elasticnet'],
    'alpha': [0.000001, 0.00001, 0.0001, 0.001],
    'l1_ratio': [0, 0.1, 0.5, 0.9, 1],
    'fit_intercept': [True, False],
    'max_iter': [5000, 10000, 25000, 50000],
    'learning_rate': ['constant', 'optimal', 'adaptive']
}

param_grid_rfr = {
    'n_estimators': [300, 400, 500],
    'max_depth': [100, 150, 200],
    'min_samples_split': [2, 5],
    'criterion': ['squared_error', 'absolute_error']
}

param_grid_gbr = {
    'loss': ['squared_error', 'absolute_error'],
    'learning_rate': [0.0001, 0.001, 0.01],
    'n_estimators': [200, 400],
    'max_depth': [1, 3, 5],
    'subsample': [0.25, 0.75]
}

param_grid_dtr = {
    'max_depth': [None, 4, 6, 8, 10],
    'criterion': ['squared_error', 'absolute_error'],
    'min_samples_split': [4, 6, 8, 10],
    'min_samples_leaf': [20, 30, 40],
    'splitter': ['best', 'random']
}


def custom_tune_regression_model_hyperparameters(model_class, X_train_scaled, y_train, 
    X_validation_scaled, y_validation, X_test_scaled, y_test, param_grid):
    
    # initialise variables to keep track of the best model & corresponding hyperparameters
    best_params = None
    best_score = -np.inf
    best_model = None

    # iterate over all possible combinations of hyperparameter values
    for param1_value in param_grid['param1']:
        for param2_value in param_grid['param2']:
            for param3_value in param_grid['param3']:

                model = model_class(param1 = param1_value, param2 = param2_value, param3 = param3_value)
                model.fit(X_train_scaled, y_train)

                score = mean_squared_error(y_validation, model.predict(X_validation_scaled))

                # update the best model if the current model has a better score
                if score > best_score:
                    best_score = score
                    best_params = {'param1': param1_value, 'param2': param2_value, 'param3': param3_value}
                    best_model = model
                    
    print(best_params)
    print(best_score)

    y_validation_pred = best_model.predict(X_validation_scaled)
    y_test_pred = best_model.predict(X_test_scaled)

    # determine performance metrics (RMSE & R2)
    validation_mse = mean_squared_error(y_validation, y_validation_pred)
    validation_rmse = np.sqrt(validation_mse)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_rmse = np.sqrt(test_mse)

    validation_r2 = r2_score(y_validation, y_validation_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    metrics = {'validation_rmse': validation_rmse, 'test_rmse': test_rmse, 'validation_r2': validation_r2, 'test_r2': test_r2}

    return best_model, best_params, metrics


def tune_regression_model_hyperparameters(model_class, X_train_scaled, y_train, 
    X_validation_scaled, y_validation, X_test_scaled, y_test, param_grid):
    
    # create instance of the GridSearchCV
    grid_search = GridSearchCV(model_class(), param_grid, cv=5)
    grid_search.fit(X_train_scaled, y_train)
    best_model = grid_search.best_estimator_

    # use the best model from the grid search to predict the validation & test sets
    y_validation_pred = grid_search.predict(X_validation_scaled)
    y_test_pred = grid_search.predict(X_test_scaled)

    # determine performance metrics (RMSE & R2)
    validation_mse = mean_squared_error(y_validation, y_validation_pred)
    validation_rmse = np.sqrt(validation_mse)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_rmse = np.sqrt(test_mse)

    validation_r2 = r2_score(y_validation, y_validation_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    metrics = {'validation_rmse': validation_rmse, 'test_rmse': test_rmse, 'validation_r2': validation_r2, 'test_r2': test_r2}

    print(metrics)
    print(grid_search.best_params_)

    return best_model, grid_search.best_params_, metrics


def save_model(model, hyperparameters, metrics, parent_folder='models/regression', model_name=None):
    
    # takes version from version.json file to append to each new model folder
    with open('version.json', 'r') as f:
        current_version = json.load(f) 

    model_class_folder = f'{parent_folder}/{model_name}'
    os.makedirs(model_class_folder, exist_ok=True)
    folder =f'{model_class_folder}/version-{current_version}'
    os.makedirs(folder, exist_ok=True)
    model_file = f'{folder}/model.joblib'
    dump(model, model_file)
    
    with open(f'{folder}/hyperparameters.json', 'w') as f:
        json.dump(hyperparameters, f)

    with open(f'{folder}/metrics.json', 'w') as f:
        json.dump(metrics, f)
    
    current_version += 1
    with open('version.json', 'w') as f:
        json.dump(current_version, f)


def evaluate_all_models():

    features, label = load_airbnb()
    X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.3)
    X_test, X_validation, y_test, y_validation = train_test_split(X_test, y_test, test_size=0.5)

    print(f'Number of samples in dataset: {len(features)}')
    print(
        'Number of samples in: '
        f'training: {len(y_train)}, '
        f'validation: {len(y_validation)}, '
        f'testing: {len(y_test)}, '
    )

    scaler = StandardScaler() # normalize the features 
    scaler.fit(X_train)

    X_train_scaled = scaler.transform(X_train)
    X_validation_scaled = scaler.transform(X_validation)
    X_test_scaled = scaler.transform(X_test)

    np.random.seed(5) # ensure each run has a level of reproducability
    model = DecisionTreeRegressor()
    for epoch in range(1000):
        model.fit(X_train_scaled, y_train)

    y_train_pred = model.predict(X_train_scaled) # training set used to optimise the model 
    y_validation_pred = model.predict(X_validation_scaled) # validation set used to make decisions about the model (which is best)
    y_test_pred = model.predict(X_test_scaled) # test set used to estimate how the model will perform on unseen (real world) data

    # evaluate model using RMSE
    train_loss = mean_squared_error(y_train, y_train_pred)
    train_rmse = np.sqrt(train_loss)

    validation_loss = mean_squared_error(y_validation, y_validation_pred)
    validation_rmse = np.sqrt(validation_loss)

    test_loss = mean_squared_error(y_test, y_test_pred)
    test_rmse = np.sqrt(test_loss)

    print(
        f'baseline_train_rmse: {train_rmse}, ' 
        f'baseline_validation_rmse: {validation_rmse}, '
        f'baseline_vest_rmse: {test_rmse}'
    )

    # evaluate model using R^2
    train_r2 = r2_score(y_train, y_train_pred)
    validation_r2 = r2_score(y_validation, y_validation_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    print(
        f'baseline_training_r2: {train_r2}, '
        f'baseline_validation_r2 {validation_r2}, '
        f'baseline_test_r2: {test_r2}'
    )

    # perform gridsearch for finding the best hyperparameters
    best_model, best_params, metrics = tune_regression_model_hyperparameters(DecisionTreeRegressor, X_train_scaled, y_train, 
        X_validation_scaled, y_validation, X_test_scaled, y_test, param_grid_dtr)

    save_model(best_model, best_params, metrics, model_name=str(type(best_model).__name__))


if __name__=='__main__':
    evaluate_all_models()