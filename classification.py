import numpy as np
import pandas as pd
from tabular_data import *
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# preprocessing steps
np.random.seed(5)
dataset = pd.read_csv('tabular_data/clean_tabular_data.csv')
features, label = load_airbnb(dataset, 'Category')

encoder = OneHotEncoder()
label_encoded = encoder.fit_transform(label.values.reshape(-1, 1))

X_train, X_test, y_train, y_test = train_test_split(features, label_encoded, test_size=0.3)
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

# get baseline classification model
model = LogisticRegression()
for epoch in range(1000):
    model.fit(X_train_scaled, y_train)

y_train_pred = model.predict(X_train_scaled) 
y_validation_pred = model.predict(X_validation_scaled)
y_test_pred = model.predict(X_test_scaled)

# evaluate model using accuracy
train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)
validation_acc = accuracy_score(y_validation, y_validation_pred)
print(
    f'baseline_train_acc: {train_acc}, '
    f'baseline_test_acc: {test_acc}, '
    f'baseline_val_acc: {validation_acc}'
)

# evaluate model using precision
train_pres = precision_score(y_train, y_train_pred)
test_pres = precision_score(y_test, y_test_pred)
validation_pres = precision_score(y_validation, y_validation_pred)
print(
    f'baseline train_pres: {train_pres}, '
    f'baseline test_pres: {test_pres}, '
    f'baseline val_pres: {validation_pres}'
)

# evaluate model using F1 score
train_f1 = f1_score(y_train, y_train_pred)
test_f1 = f1_score(y_test, y_test_pred)
validation_f1 = f1_score(y_validation, y_validation_pred)
print(
    f'baseline_train_f1: {train_f1}, '
    f'baseline test_f1: {test_f1}, '
    f'baseline_val_f1: {validation_f1}'
)

# evaluate model using recall 
train_recall = recall_score(y_train, y_train_pred)
test_recall = recall_score(y_test, y_test_pred)
validation_recall = recall_score(y_validation, y_validation_pred)
print(
    f'baseline_train_recall: {train_recall}, '
    f'baseline_test_recall: {test_recall}, '
    f'baseline_val_recall: {validation_recall}'
)