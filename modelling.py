from tabular_data import *
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error


features, label = load_airbnb()

# use 80% of data for training and 20% of the data for testing
# random_state ensures same random samples of data will be used for training & testing each time code is run
X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.2, random_state=42)

print(f'Number of samples in datase: {len(features)}')
print('Number of samples in:')
print(f'    training: {len(X_train)}')
print(f'    testing: {len(X_test)}')


model = SGDRegressor()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)



mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
