import json
import joblib
import os
import numpy as np
import pandas as pd
from tabular_data import *
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# preprocessing steps
np.random.seed(5)
dataset = pd.read_csv('tabular_data/clean_tabular_data.csv')
features, label = load_airbnb(dataset, 'Category')
label_series = dataset['Category']


encoder = LabelEncoder()
label_categories = label_series.unique()
label_encoded = encoder.fit_transform(label_series)


X_train, X_test, y_train, y_test = train_test_split(features, label_encoded, test_size=0.3)
X_test, X_validation, y_test, y_validation = train_test_split(X_test, y_test, test_size=0.5)

print(f'Number of samples in dataset: {len(features)}')
print(
    'Number of samples in: '
    f'training: {y_train.shape[0]}, '
    f'validation: {y_validation.shape[0]}, '
    f'testing: {y_test.shape[0]}, '
)

# normalize the features 
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_validation_scaled = scaler.transform(X_validation)
X_test_scaled = scaler.transform(X_test)

# get baseline classification model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

y_train_pred = model.predict(X_train_scaled) 
y_validation_pred = model.predict(X_validation_scaled)
y_test_pred = model.predict(X_test_scaled)

# evaluate model using accuracy
train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)
validation_acc = accuracy_score(y_validation, y_validation_pred)
print(
    f'baseline_train_acc: {round(train_acc, 4)}, '
    f'baseline_test_acc: {round(test_acc, 4)}, '
    f'baseline_val_acc: {round(validation_acc, 4)}'
)

# evaluate model using precision
train_pres = precision_score(y_train, y_train_pred, average='macro')
test_pres = precision_score(y_test, y_test_pred, average='macro')
validation_pres = precision_score(y_validation, y_validation_pred, average='macro')
print(
    f'baseline train_pres: {round(train_pres, 4)}, '
    f'baseline test_pres: {round(test_pres, 4)}, '
    f'baseline val_pres: {round(validation_pres, 4)}'
)

# evaluate model using F1 score
train_f1 = f1_score(y_train, y_train_pred, average='macro')
test_f1 = f1_score(y_test, y_test_pred, average='macro')
validation_f1 = f1_score(y_validation, y_validation_pred, average='macro')
print(
    f'baseline_train_f1: {round(train_f1, 4)}, '
    f'baseline test_f1: {round(test_f1, 4)}, '
    f'baseline_val_f1: {round(validation_f1, 4)}'
)

# evaluate model using recall 
train_recall = recall_score(y_train, y_train_pred, average='macro')
test_recall = recall_score(y_test, y_test_pred, average='macro')
validation_recall = recall_score(y_validation, y_validation_pred, average='macro')
print(
    f'baseline_train_recall: {round(train_recall, 4)}, '
    f'baseline_test_recall: {round(test_recall, 4)}, '
    f'baseline_val_recall: {round(validation_recall, 4)}'
)


def tune_regression_model_hyperparameters(model_class, X_train_scaled, y_train, 
    X_validation_scaled, y_validation, X_test_scaled, y_test, parameter_grid):
    np.random.seed(5)

    models = {
        'LogisticRegression': LogisticRegression,
        'RandomForestClassifier': RandomForestClassifier,
        'DecisionTreeClassifier': DecisionTreeClassifier,
        'GradientBoostingClassifier': GradientBoostingClassifier
    }

    # perform grid search to find best hyperparameters
    model = models[model_class]()
    grid_search = GridSearchCV(estimator=model, param_grid=parameter_grid, scoring='accuracy')
    grid_search.fit(X_train_scaled, y_train)
    best_model = models[model_class](**grid_search.best_params_)
    best_model.fit(X_train_scaled, y_train)

    # use the best model from the grid search to predict the validation & test sets
    y_validation_pred = best_model.predict(X_validation_scaled)
    y_test_pred = best_model.predict(X_test_scaled)

    # determine performance metrics
    validation_acc = accuracy_score(y_validation, y_validation_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    validation_f1 = f1_score(y_validation, y_validation_pred, average='macro')
    test_f1 = f1_score(y_test, y_test_pred, average='macro')

    metrics = {'validation_acc': validation_acc, 'test_acc': test_acc, 'validation_f1': validation_f1, 'test_f1': test_f1}
    best_model_data = [best_model, grid_search.best_params_, metrics]
    print(best_model_data)

    return best_model_data


def save_model(best_model_data, model_name):
    
    model, hyperparameters, metrics = best_model_data

    model_folder = 'models/classification/' + model_name
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    
    joblib.dump(model, f'{model_folder}/model.joblib')

    with open(f'{model_folder}/hyperparameters.json', 'w') as file:
        json.dump(hyperparameters, file)

    with open(f'{model_folder}/metrics.json', 'w') as file:
        json.dump(metrics, file)


def evaluate_all_models():

    np.random.seed(5)

    lr_model = tune_regression_model_hyperparameters('LogisticRegression',  X_train_scaled, y_train, 
    X_validation_scaled, y_validation, X_test_scaled, y_test, parameter_grid =
    {
    'C': [0.001, 0.01, 0.1, 1, 10],
    'penalty': ['l2'],
    'max_iter': [100, 500, 1000],
    'multi_class': ['ovr'],
    'solver': ['lbfgs']   
    })

    save_model(lr_model, 'LogisticRegression')

    rfc_model = tune_regression_model_hyperparameters('RandomForestClassifier',  X_train_scaled, y_train, 
    X_validation_scaled, y_validation, X_test_scaled, y_test, parameter_grid = 
    {
    'n_estimators': [50, 75, 100],
    'criterion': ['gini', 'entropy'],
    'max_depth': [30, 40, 50],
    'min_samples_split': [2, 0.1, 0.2],
    'min_samples_leaf': [1, 2, 3]
    })

    save_model(rfc_model, 'RandomForestClassifier')

    gbc_model = tune_regression_model_hyperparameters('GradientBoostingClassifier',  X_train_scaled, y_train, 
    X_validation_scaled, y_validation, X_test_scaled, y_test, parameter_grid = 
    {
    'n_estimators': [25, 50, 100,],
    'learning_rate': [0.01, 0.1, 1],
    'max_depth': [1, 3, 5],
    'max_features': [1, 2, 3]
    })

    save_model(gbc_model, 'GradientBoostingClassifier')

    dtc_model = tune_regression_model_hyperparameters('GradientBoostingClassifier',  X_train_scaled, y_train, 
    X_validation_scaled, y_validation, X_test_scaled, y_test, parameter_grid = 
    {
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 4, 0.2, 0.4],
    'min_samples_leaf': [1, 3, 5],
    'max_features': [4, 6, 8]
    })

    save_model(dtc_model, 'DecisionTreeClassifier')


def find_best_model(models_directory):
    best_model = None
    best_acc = -float('inf')
    best_f1 = float('inf')

    for model_name in os.listdir(models_directory):
        metrics_path = os.path.join(models_directory, model_name, 'metrics.json')
        with open(metrics_path) as f:
            metrics = json.load(f)
            val_acc = metrics['validation_acc'] 
            val_f1 = metrics['validation_f1']

            if val_acc > best_acc and val_f1 > best_f1:
                best_model = model_name
                best_acc = val_acc
                best_f1 = val_f1
    
    return best_model

if __name__=='__main__':
    evaluate_all_models()
    best_model = find_best_model('models/classification')
    print(best_model)
    