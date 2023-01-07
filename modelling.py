from tabular_data import *
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error


features, label = load_airbnb()

# use 80% of data for training and 20% of the data for testing
X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.2)

# validation set used to compare models
X_test, X_validation, y_test, y_validation = train_test_split(X_test, y_test, test_size=0.2)

print(f'Number of samples in datase: {len(features)}')
print('Number of samples in:')
print(f'    training: {len(y_train)}')
print(f'    validation: {len(y_validation)}')
print(f'    testing: {len(y_test)}')

model = SGDRegressor()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)



mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print(mse)
print(mae)