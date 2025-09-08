''' Problem: Predicting Energy Consumption in a Smart Industrial Facility
Context
A medium-sized industrial plant has installed a smart energy monitoring
system using IoT sensors across the facility. These sensors measure:

- Electricity consumption per machine (kWh)

- Temperature and humidity of production zones

- Production load (number of units produced per hour)

- Operational hours per machine

The facility manager wants to optimize energy usage and reduce costs by
understanding how different factors affect total electricity consumption.

Challenge
The goal is to predict daily energy consumption based on operational and
environmental parameters so that:

- Managers can schedule production optimally to avoid peak energy costs.

- Predictive alerts can be generated if energy usage is likely to exceed
thresholds.

- Energy efficiency initiatives can be evaluated quantitatively.

Why Linear Regression is Ideal
- Target Variable is Continuous: Daily energy consumption (kWh) is continuous,
which suits regression models.

- Linear Relationships Exist: Energy consumption tends to increase linearly
with operational hours, production load, and number of active machines.

- Environmental factors (temperature, humidity) have predictable, roughly
linear effects on HVAC energy usage.

- Interpretability is Crucial: Facility engineers and managers need to
understand how each factor contributes to energy consumption.

- Coefficients from linear regression can directly show, for example: “Each
additional unit produced adds 2.5 kWh” “Each 1°C increase in temperature adds
0.8 kWh due to HVAC load”

- Computationally Efficient: Can run in real-time on Edge AI devices, such as
microcontrollers or small industrial servers, for near-instant predictions.

Baseline and Benchmark: Linear regression provides a simple, reliable baseline
before experimenting with more complex AI models (e.g., neural networks or
gradient boosting).

Expected Outcomes
- A trained linear regression model predicts daily energy consumption based on
production and environmental conditions.

- The facility can plan operations to minimize energy costs while meeting
production targets.

- Engineers can identify key contributors to energy waste (e.g., inefficient
machines or excessive HVAC usage).

- This problem perfectly combines: AIoT / Edge AI, IoT sensor data, industrial
automation, and energy optimization, while Linear Regression remains the
optimal first-step solution due to its interpretability, simplicity, and 
efficiency.

'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Python Exercise: Predicting Energy Consumption in a Smart Industrial Facility

'''Scenario Recap
You are tasked with predicting daily energy consumption (kWh) in an
industrial plant using IoT sensor data:

Features (X): Machine Load, Units Produced, Temperature Zone 1,
Temperature Zone 2, Humidity

Target (y): Daily Energy Consumption (kWh)'''

# Step 1 – Generate Synthetic IoT Data

# Seed for reproducibility
np.random.seed(42)

# Generate synthetic IoT dataset
n_days = 300

machine_load = np.random.uniform(0.5, 1.0, n_days)  # Operational load (50%-100%)
units_produced = np.random.randint(80, 200, n_days)  # Units produced per day
temp_zone1 = np.random.uniform(20, 30, n_days)  # Temperature in zone 1
temp_zone2 = np.random.uniform(20, 30, n_days)  # Temperature in zone 2
humidity = np.random.uniform(40, 60, n_days)  # Humidity percentage

# Energy consumption formula with some noise
noise = np.random.normal(0, 10, n_days)
energy_kwh = (
    200 + 
    (machine_load * 300) + 
    (units_produced * 2.5) + 
    (temp_zone1 * 5) + 
    (temp_zone2 * 4.5) + 
    (humidity * 1.2) + 
    noise
)

# Create DataFrame
data = pd.DataFrame({
    "Machine_Load": machine_load,
    "Units_Produced": units_produced,
    "Temp_Zone1": temp_zone1,
    "Temp_Zone2": temp_zone2,
    "Humidity": humidity,
    "Energy_kWh": energy_kwh
})

print(data.head())

# Step 2 – Data Preprocessing
# Features and target
X = data[["Machine_Load", "Units_Produced", "Temp_Zone1", "Temp_Zone2",
          "Humidity"]]
y = data["Energy_kWh"]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=42)

# Feature scaling (important for IoT/Edge AI deployment)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# Step 3 – Train Linear Regression Model
# Create and train the model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)
# Step 4 – Evaluate the Model
# Calculate metrics
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Root Mean Squared Error (RMSE):", round(rmse, 2))
print("Mean Absolute Error (MAE):", round(mae, 2))
print("R² Score:", round(r2, 3))
# Step 5 – Interpret Coefficients
# Model coefficients
coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})

intercept = model.intercept_

print("Intercept:", round(intercept, 2))
print(coefficients)
'''Interpretation Example:

Each unit increase in Units_Produced adds roughly 2.5 kWh to energy
consumption.
Each 1°C increase in Temp_Zone1 adds ~5 kWh due to HVAC load.
Machine_Load coefficient indicates how operational intensity affects total
consumption.'''

# Step 6 – Visualize Actual vs Predicted
plt.scatter(y_test, y_pred, alpha=0.7, color="blue")
plt.plot([y.min(), y.max()], [y.min(), y.max()], "r--", lw=2)
plt.xlabel("Actual Energy Consumption (kWh)")
plt.ylabel("Predicted Energy Consumption (kWh)")
plt.title("Actual vs Predicted Energy Consumption")
plt.show()
'''✅ Exercise Extensions
Add new IoT variables: e.g., ambient light, vibration, or machine age.
Try regularization: Ridge or Lasso regression to improve stability and
reduce overfitting.

Deploy on Edge AI devices: Use this model for real-time energy predictions
in the factory.

Feature importance analysis: Rank features by impact on energy consumption.

This exercise simulates a real-world industrial IoT scenario and gives
hands-on experience in data preprocessing, model training, evaluation, and
interpretation.'''
