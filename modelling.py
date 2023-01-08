import numpy as np
from tabular_data import *
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error


features, label = load_airbnb()
X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.3)
X_test, X_validation, y_test, y_validation = train_test_split(X_test, y_test, test_size=0.5)

print(f'Number of samples in dataset: {len(features)}')
print('Number of samples in:')
print(f'    training: {len(y_train)}')
print(f'    validation: {len(y_validation)}')
print(f'    testing: {len(y_test)}')


# ensure each run has a level of reproducability
np.random.seed(2)

model = SGDRegressor()

for epoch in range(1000):
    model.fit(X_train, y_train)

y_train_pred = model.predict(X_train) # training set used to optimise the model 
y_validation_pred = model.predict(X_validation) # validation set used to make decisions about the model (which is best)
y_test_pred = model.predict(X_test) # test set used estimate how the model will perform on unseen (real wordl) data

# check for data leakage
train_loss = mean_squared_error(y_train, y_train_pred)
validation_loss = mean_squared_error(y_validation, y_validation_pred)
test_loss = mean_squared_error(y_test, y_test_pred)

print(
    f'Train loss: {train_loss}, ' 
    f'Validation loss: {validation_loss}, '
    f'Test loss: {test_loss}'
)