import numpy as np
import pandas as pd
from tabular_data import *
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder

# preprocessing steps
np.random.seed(5)
dataset = pd.read_csv('tabular_data/clean_tabular_data.csv')
features, label = load_airbnb(dataset, 'Category')

encoder = OneHotEncoder()
label_encoded = encoder.fit_transform(label.values.reshape(-1, 1))