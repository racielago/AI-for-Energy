# Predicting electricity consumption (kWh) in a small factory based on the
# number of working machines and the average working hours per day.

# This small Python script creates a synthetic energy dataset, fits a linear
# regression model to predict daily electricity consumption from two inputs
# (number of machines and working hours), prints the learned model parameters, 
# makes predictions for two new scenarios, and plots actual vs predicted
# consumption to visually check the fit.

# Create a synthetic dataset
# We’ll generate synthetic data that follows a linear relationship with
# some noise:
# Energy Consumption=50+20×(Machines)+10×(Hours)+Noise

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# -----------------------------------------------------------------------------

# Step 1: Create synthetic dataset

# -----------------------------------------------------------------------------

# Create reproducible randomness

np.random.seed(42)
# Sets the random number generator to a fixed state so every time you run
# the script you get the same random numbers. This helps debugging and makes
# results reproducible. 42 is just a chosen seed value (any integer works).

# Generate features (inputs)

machines = np.random.randint(5, 20, 50) # between 5 and 20 machines

hours = np.random.randint(4, 12, 50)  # between 4 and 12 hours/day

# np.random.randint(low, high, size) returns size integers drawn uniformly
# from low (inclusive) to high (exclusive).

# Generate target (energy consumption) with noise

consumption = 50 + 20 * machines + 10 * hours + np.random.normal(0, 15, 50)

# This line builds the synthetic target variable

# baseline 50 (a constant offset)
# + 20 * machines means each additional machine adds about 20 kWh,
# + 10 * hours means each extra working hour adds about 10 kWh,
# + np.random.normal(0, 15, 50) adds Gaussian (normal) noise with mean 0 and
# standard deviation 15 to each sample.
# np.random.normal(0, 15, 50) returns an array of 50 floats sampled from that
# normal distribution — this simulates measurement error or unobserved
# variability in real life.
# The final consumption is a NumPy array of length 50 with float values (kWh).

# Build a Pandas DataFrame and print first rows

data = pd.DataFrame({
    "Machines": machines,
    "Hours": hours,
    "Consumption_kWh": consumption
    })

print(data.head())

# pd.DataFrame({...}) creates a table (DataFrame) with three columns:
# "Machines", "Hours", and "Consumption_kWh". Each column has 50 rows.

# data.head() returns the first 5 rows of the DataFrame. print() shows that on
# the console so you can inspect the generated samples quickly.

# -----------------------------------------------------------------------------

# Step 2: Train linear regression model

# -----------------------------------------------------------------------------

# Select features X and target y

X = data[["Machines", "Hours"]]   # independent variables

y = data["Consumption_kWh"]       # dependent variable

# X is a DataFrame of shape (50, 2) containing the two input columns.
# Note the double brackets data[["col1","col2"]] — that returns a
# DataFrame (2D).
# y is a Pandas Series (1D) containing the target values (shape (50,)).
# sklearn expects X to be 2D (samples × features) and y to be 1D (targets).

# Create model and fit it

model = LinearRegression()
model.fit(X, y)

# model = LinearRegression() creates an instance of the linear regression
# algorithm with default settings (it will fit an intercept by default).

# model.fit(X, y) computes the best-fitting parameters
# (intercept and coefficients) using ordinary least squares:
# it finds coefficients that minimize the sum of squared differences
# between predicted and true y values.
# Under the hood this uses linear algebra (e.g., singular value 
# decomposition or normal equations) to solve for coefficients.

# -----------------------------------------------------------------------------

# Step 3: View learned parameters

# -----------------------------------------------------------------------------

print("Intercept (baseline):", model.intercept_)
print("Coefficients:", model.coef_)

# model.intercept_ is the learned constant term (the baseline consumption
# when all features are zero).
# model.coef_ is a NumPy array of length 2 with the slope for each feature,
# in the same order as X columns you passed (["Machines","Hours"]).
# Example interpretation: if model.coef_ = [19.8, 9.9] and
# intercept_ = 51.0, then predicted
# consumption ≈ 51.0 + 19.8*Machines + 9.9*Hours.

# -----------------------------------------------------------------------------

# Step 4: Make predictions for new scenarios

# -----------------------------------------------------------------------------

new_data = pd.DataFrame({
    "Machines": [10, 15],
    "Hours": [8, 10]
    })

predictions = model.predict(new_data)
print("/nPredictions for new scenarios: ")
print(new_data.assign(Predicted_Consumption=predictions))

# new_data is a small DataFrame with two hypothetical cases:
# row 1: 10 machines, 8 hours
# row 2: 15 machines, 10 hours
# model.predict(new_data) returns an array of predicted consumption values
# for those cases, using the learned coefficients.
# new_data.assign(Predicted_Consumption=predictions) creates a copy of
# new_data and adds a new column Predicted_Consumption with the numeric
# predictions; print() shows them.
# Important note: scikit-learn uses the order of features when converting
# DataFrame to numeric arrays. Make sure new_data columns are in the same
# order or have the same structure as X used to train the model — otherwise
# predictions will be wrong.
