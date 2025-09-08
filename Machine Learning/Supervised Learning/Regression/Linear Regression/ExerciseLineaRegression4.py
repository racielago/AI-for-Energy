'''Python Exercise: Time-Series Forecasting of Energy Consumption in a Smart
Industrial Facility
Scenario Recap
You have hourly IoT sensor data from an industrial facility, including:

- Machine load

- Units produced

- Temperature Zone 1 & 2

- Humidity

- The goal is to predict the next hour’s energy consumption using Linear
Regression and lagged features (past values as predictors).'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Step 1 – Generate Synthetic Time-Series IoT Data

# Seed for reproducibility
np.random.seed(42)

# Generate hourly data for 30 days (720 hours)
hours = 720
time_index = pd.date_range(start="2025-09-01", periods=hours, freq="h")

# Simulate sensor readings
machine_load = np.random.uniform(0.5, 1.0, hours)  # Operational load
units_produced = np.random.randint(5, 20, hours)   # Units per hour
temp_zone1 = np.random.uniform(20, 30, hours)      # Temp Zone1
temp_zone2 = np.random.uniform(20, 30, hours)      # Temp Zone2
humidity = np.random.uniform(40, 60, hours)        # Humidity

# Simulate energy consumption with noise
noise = np.random.normal(0, 5, hours)
energy_kwh = (
    20 +
    (machine_load * 30) +
    (units_produced * 2.5) +
    (temp_zone1 * 1.5) +
    (temp_zone2 * 1.2) +
    (humidity * 0.8) +
    noise
)

# Create DataFrame
data = pd.DataFrame({
    "Timestamp": time_index,
    "Machine_Load": machine_load,
    "Units_Produced": units_produced,
    "Temp_Zone1": temp_zone1,
    "Temp_Zone2": temp_zone2,
    "Humidity": humidity,
    "Energy_kWh": energy_kwh
})

data.set_index("Timestamp", inplace=True)
print(data.head())
# Step 2 – Create Lagged Features for Forecasting
# We’ll use the previous hour’s energy and sensor readings to predict the
# next hour:

# Create lagged features
data["Energy_kWh_Lag1"] = data["Energy_kWh"].shift(1)
data["Machine_Load_Lag1"] = data["Machine_Load"].shift(1)
data["Units_Produced_Lag1"] = data["Units_Produced"].shift(1)

# Drop first row with NaN due to lag
data = data.dropna()

# Features and target
X = data[["Energy_kWh_Lag1", "Machine_Load_Lag1", "Units_Produced_Lag1",
          "Temp_Zone1", "Temp_Zone2", "Humidity"]]
y = data["Energy_kWh"]
# Step 3 – Split Data into Training and Testing
# Use first 80% for training, last 20% for testing (time-series split)
split_index = int(len(data) * 0.8)
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# Step 4 – Train Linear Regression Model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)
# Step 5 – Evaluate the Model
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("RMSE:", round(rmse, 2))
print("R² Score:", round(r2, 3))
# Step 6 – Visualize Forecast vs Actual
plt.figure(figsize=(12, 6))
plt.plot(y_test.index, y_test, label="Actual Energy")
plt.plot(y_test.index, y_pred, label="Predicted Energy", alpha=0.7)
plt.xlabel("Time")
plt.ylabel("Energy Consumption (kWh)")
plt.title("Hourly Energy Consumption Forecast")
plt.legend()
plt.show()
# Step 7 – Interpret Model Coefficients
coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})
print("Intercept:", round(model.intercept_, 2))
print(coefficients)
'''Interpretation Example:

Energy_kWh_Lag1: Shows dependency on previous hour’s energy (temporal pattern).

Machine_Load_Lag1 and Units_Produced_Lag1: Capture operational effect from the
last hour.

Temperature and humidity coefficients indicate environmental influence on
energy consumption.

✅ Extensions for Edge AI / Industrial IoT
Predict next hour or next day energy for proactive energy management.

Deploy model on Edge AI devices (e.g., industrial controllers or microservers)
for real-time forecasting.

Add rolling averages or multi-hour lags to improve forecast accuracy.

Combine with alert systems to warn when predicted consumption exceeds
thresholds.

Extend to multivariate forecasting across multiple machines or zones.'''
