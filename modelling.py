from tabular_data import *
import sklearn

features, labels = load_airbnb('Price_Night')


# Create an instance of the SGDRegressor class
sgd_regressor = SGDRegressor(loss='squared_loss')

# Fit the model to the training data
sgd_regressor.fit(features, labels)


# Use the model to make predictions on new data
predictions = sgd_regressor.predict(new_data)
