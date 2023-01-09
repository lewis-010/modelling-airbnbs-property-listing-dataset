import numpy as np
from tabular_data import *
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
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


# normalize the features 
scaler = StandardScaler()
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_validation_scaled = scaler.transform(X_validation)
X_test_scaled = scaler.transform(X_test)


np.random.seed(5) # ensure each run has a level of reproducability

model = SGDRegressor()

for epoch in range(1000):
    model.fit(X_train_scaled, y_train)

y_train_pred = model.predict(X_train_scaled) # training set used to optimise the model 
y_validation_pred = model.predict(X_validation_scaled) # validation set used to make decisions about the model (which is best)
y_test_pred = model.predict(X_test_scaled) # test set used to estimate how the model will perform on unseen (real world) data


# evaluate model
train_loss = mean_squared_error(y_train, y_train_pred)
train_rmse = np.sqrt(train_loss)

validation_loss = mean_squared_error(y_validation, y_validation_pred)
validation_rmse = np.sqrt(validation_loss)

test_loss = mean_squared_error(y_test, y_test_pred)
test_rmse = np.sqrt(test_loss)

print(
    f'Train_rmse: {train_rmse}, ' 
    f'Validation_rmse: {validation_rmse}, '
    f'Test_rmse: {test_rmse}'
)
