import numpy as np
import pandas as pd
from tabular_data import *
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

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
    X_validation_scaled, y_validation, X_test_scaled, y_test, param_grid):
    np.random.seed(2)

