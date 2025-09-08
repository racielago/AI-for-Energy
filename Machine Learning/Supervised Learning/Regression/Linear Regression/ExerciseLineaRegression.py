# Practice Exercise: Predicting Energy Consumption with Linear Regression
# Scenario
# An energy company wants to predict the electricity consumption (kWh) of
# households based on several factors, such as temperature, number of
# residents, and household size. You will use Linear Regression to model the
# relationship between these variables.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 1 – Generate Data

# Random seed for reproducibility
np.random.seed(42)

# Generate synthetic dataset
n_samples = 200
temperature = np.random.uniform(0, 35, n_samples)  # between 0 and 35°C
residents = np.random.randint(1, 9, n_samples)  # between 1 and 5 residents
house_size = np.random.uniform(40, 200, n_samples)  # between 40 and 200 m²

# Define the target with some noise
noise = np.random.normal(0, 5, n_samples)                  # random noise
consumption = 20 + (2.5 * residents) + (0.1 * house_size) - (1.2 * temperature)
+ noise

# Create DataFrame
data = pd.DataFrame({
    "temperature": temperature,
    "residents": residents,
    "house_size": house_size,
    "consumption": consumption
})

print(data.head())

# Step 2 – Train the Linear Regression Model
# Features (X) and Target (y)
X = data[["temperature", "residents", "house_size"]]
y = data["consumption"]

# Split data: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Step 3 – Evaluate the Model
# Metrics
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("Root Mean Squared Error (RMSE):", rmse)
print("R² Score:", r2)

# Model coefficients
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)
print("Features:", X.columns.tolist())

# Step 4 – Visualize
plt.scatter(y_test, y_pred, alpha=0.7, color="blue")
plt.xlabel("Actual Consumption (kWh)")
plt.ylabel("Predicted Consumption (kWh)")
plt.title("Actual vs Predicted Energy Consumption")
plt.plot([y.min(), y.max()], [y.min(), y.max()], "r--")
plt.show()
