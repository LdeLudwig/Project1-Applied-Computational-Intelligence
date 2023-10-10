# Computational Intelligence - Project 1b - Neural Network
# Lucas Xavier and Simon Nygren
# 2023-10-06

import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# TestMe
TestMe = pd.read_csv('Proj1_TestS.csv', sep=',', decimal='.')

# csv reading
dataTrain = pd.read_csv('fuzzy_dataset.csv')

X = dataTrain.drop(columns=['CLPVariation']).values  # Input features (all columns except CLPVariation) train
y = dataTrain['CLPVariation'].values  # Output train(CLPVariation column)

# Split the dataset into training and testing sets, training set 70 percent
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Split the dataset into validation and testing, validation and test set 15 percent each
X_validation, X_test, y_validation, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=21)


# Instantiate MLPRegressor
nn = MLPRegressor(
    activation='relu',
    hidden_layer_sizes=(16, 32),
    alpha=0.00001,
    solver='adam',
    random_state=42,
    early_stopping=False,
    max_iter=2000
)

# Train the model
nn.fit(X_train, y_train)

# Make predictions of test dataset
pred = nn.predict(X_test)

validation_set_rsquared = nn.score(X_validation, y_validation)
test_set_rsquared = nn.score(X_test, y_test)
test_set_rmse = np.sqrt(mean_squared_error(y_test, pred))

# Print R_squared and RMSE value
print('R_squared_validation value', validation_set_rsquared)
print('R_squared value: ', test_set_rsquared)
print('RMSE: ', test_set_rmse)

# Create a csv file with CLPVariation results of NN
prediction = nn.predict(TestMe)
dfTestResult = pd.read_csv('TestResult.csv')
dfTestResult['CLPVariation_NN'] = prediction
dfTestResult.to_csv('TestResult.csv', index=False)


