import numpy as np
from tabular_data import *
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

features, label = load_airbnb()
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


np.random.seed(5) # ensure each run has a level of reproducability
model = SGDRegressor()
for epoch in range(1000):
    model.fit(X_train_scaled, y_train)

y_train_pred = model.predict(X_train_scaled) # training set used to optimise the model 
y_validation_pred = model.predict(X_validation_scaled) # validation set used to make decisions about the model (which is best)
y_test_pred = model.predict(X_test_scaled) # test set used to estimate how the model will perform on unseen (real world) data


# evaluate model using RMSE
train_loss = mean_squared_error(y_train, y_train_pred) # should I be evaluating using the training set??
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

# evaluate model using R^2
train_r2 = r2_score(y_train, y_train_pred) # should I be evaluating using the training set??
validation_r2 = r2_score(y_validation, y_validation_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(
    f'Training_r2: {train_r2}, '
    f'Validation_r2 {validation_r2}, '
    f'Test_r2: {test_r2}'
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

    # determine performance metrics
    validation_mse = mean_squared_error(y_validation, y_validation_pred)
    validation_rmse = np.sqrt(validation_mse)

    test_mse = mean_squared_error(y_test, y_test_pred)
    test_rmse = np.sqrt(test_mse)
    
