import json
import joblib
import numpy as np
import pandas as pd
import os
from tabular_data import *
from sklearn.linear_model import SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split

# preprocessing steps
np.random.seed(5)
dataset = pd.read_csv('tabular_data/clean_tabular_data.csv')
features, label = load_airbnb(dataset, 'Price_Night')
X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.3)
X_test, X_validation, y_test, y_validation = train_test_split(X_test, y_test, test_size=0.5)

print(f'Number of samples in dataset: {len(features)}')
print(
    'Number of samples in: '
    f'training: {len(y_train)}, '
    f'validation: {len(y_validation)}, '
    f'testing: {len(y_test)}, '
)

# normalize the features 
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_validation_scaled = scaler.transform(X_validation)
X_test_scaled = scaler.transform(X_test)

# get baseline regression model
model = SGDRegressor()
for epoch in range(1000):
    model.fit(X_train_scaled, y_train)

y_train_pred = model.predict(X_train_scaled) 
y_validation_pred = model.predict(X_validation_scaled)
y_test_pred = model.predict(X_test_scaled)

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
    f'baseline_test_rmse: {test_rmse}'
)

# evaluate model using R2
train_r2 = r2_score(y_train, y_train_pred)
validation_r2 = r2_score(y_validation, y_validation_pred)
test_r2 = r2_score(y_test, y_test_pred)
print(
    f'baseline_training_r2: {train_r2}, '
    f'baseline_validation_r2 {validation_r2}, '
    f'baseline_test_r2: {test_r2}'
)


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
    X_validation_scaled, y_validation, X_test_scaled, y_test, parameter_grid):
    
    models = {
        'SDGRegressor': SGDRegressor,
        'DecisionTreeRegressor': DecisionTreeRegressor,
        'GradientBoostingRegressor': GradientBoostingRegressor,
        'RandomForestRegressor': RandomForestRegressor
    }

    # perform grid search to find best hyperparameters
    model = models[model_class]()
    grid_search = GridSearchCV(estimator=model, param_grid=parameter_grid, scoring=['r2', 'neg_root_mean_squared_error'], refit='neg_root_mean_squared_error', cv=5)
    grid_search.fit(X_train_scaled, y_train)
    best_model = models[model_class](**grid_search.best_params_)
    best_model.fit(X_train_scaled, y_train)

    # use the best model from the grid search to predict the validation & test sets
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
    best_model_data = [best_model, grid_search.best_params_, metrics]
    print(best_model_data)

    return best_model_data


def save_model(best_model_data, model_name):
    
    model, hyperparameters, metrics = best_model_data

    model_folder = 'models/regression/' + model_name
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    
    joblib.dump(model, f'{model_folder}/model.joblib')

    with open(f'{model_folder}/hyperparameters.json', 'w') as file:
        json.dump(hyperparameters, file)

    with open(f'{model_folder}/metrics.json', 'w') as file:
        json.dump(metrics, file)
    

def evaluate_all_models():

    np.random.seed(5) 

    sgd_model = tune_regression_model_hyperparameters('SDGRegressor', X_train_scaled, y_train, 
        X_validation_scaled, y_validation, X_test_scaled, y_test, parameter_grid = 
    {
    'loss': ['squared_error', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'],
    'penalty': ['l2', 'l1', 'elasticnet'],
    'alpha': [0.000001, 0.00001, 0.0001, 0.001],
    'l1_ratio': [0, 0.1, 0.5, 0.9, 1],
    'fit_intercept': [True, False],
    'max_iter': [5000, 10000, 25000, 50000],
    'learning_rate': ['constant', 'optimal', 'adaptive']
    })

    save_model(sgd_model, 'SDGRegressor')

    rfr_model = tune_regression_model_hyperparameters('RandomForestRegressor', X_train_scaled, y_train, 
        X_validation_scaled, y_validation, X_test_scaled, y_test, parameter_grid = 
    {
    'n_estimators': [50, 100, 150],
    'criterion': ['squared_error', 'absolute_error'],
    'max_depth': [30, 40, 50],
    'min_samples_split': [2, 0.1, 0.2],
    'max_features': [1, 2]
    })

    save_model(rfr_model, 'RandomForestRegressor')

    dtr_model = tune_regression_model_hyperparameters('DecisionTreeRegressor', X_train_scaled, y_train, 
        X_validation_scaled, y_validation, X_test_scaled, y_test, parameter_grid = 
    {
    'criterion': ['squared_error', 'absolute_error'],
    'max_depth': [15, 30, 45, 60],
    'min_samples_split': [2, 4, 0.2, 0.4],
    'max_features': [4, 6, 8]
    })

    save_model(dtr_model, 'DecisionTreeRegressor')

    gbr_model = tune_regression_model_hyperparameters('GradientBoostingRegressor', X_train_scaled, y_train, 
        X_validation_scaled, y_validation, X_test_scaled, y_test, parameter_grid = 
    {
    'n_estimators': [25, 50, 100],
    'loss': ['squared_error', 'absolute_error'],
    'max_depth': [1, 3, 5],
    'learning_rate': [0.05, 0.1, 0.2],
    'max_features': [1, 2, 3]
    })

    save_model(gbr_model, 'GradientBoostingRegressor')


def find_best_model(models_directory):
    best_model = None
    best_r2 = -float('inf')
    best_rmse = float('inf')

    for model_name in os.listdir(models_directory):
        metrics_path = os.path.join(models_directory, model_name, 'metrics.json')
        with open(metrics_path) as f:
            metrics = json.load(f)
            val_r2 = metrics['validation_r2'] 
            val_rmse = metrics['validation_rmse']

            if val_r2 > best_r2 and val_rmse < best_rmse:
                best_model = model_name
                best_r2 = val_r2
                best_rmse = val_rmse
    
    return best_model


if __name__=='__main__':
    evaluate_all_models()
    best_model = find_best_model('models/regression')
    print(best_model)