# 🔥 Advanced Exercise: Linear Regression with Real Energy Data

'''Objective
Use a real-world dataset to build, train, and evaluate a linear regression
model that predicts energy consumption or energy efficiency.

Step 1. Choose a Dataset
You can download one of these datasets (both are well-known in energy-related
research):

1) Energy Efficiency Dataset (UCI Machine Learning Repository)
- Source: UCI Energy Efficiency Dataset
- Description: Contains building characteristics (walls, roof area, glazing,
orientation, etc.) and two target variables: Heating Load and Cooling Load.
- Goal: Predict heating or cooling load from building features.

2) Household Electric Power Consumption Dataset (Kaggle)
- Source: Household Power Consumption on Kaggle
- Description: Records electric power consumption in one household with a
1-minute sampling rate over 4 years.
- Goal: Predict Global_active_power from other variables like Voltage,
Global_reactive_power, Sub_metering values, etc.'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Step 2. Load the Dataset

# Example with UCI Energy Efficiency dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx"
df = pd.read_excel(url)

print(df.head())

# Step 3. Data Preprocessing
# Check for missing values

print(df.isnull().sum())
# Rename columns for clarity (if dataset uses generic names like X1, X2...).

df.columns = [
    "Relative_Compactness", "Surface_Area", "Wall_Area", "Roof_Area",
    "Overall_Height", "Orientation", "Glazing_Area", "Glazing_Area_Distribution",
    "Heating_Load", "Cooling_Load"
]

# Select features (X) and target (y)
# Example: Predict heating load.

X = df.drop(columns=["Heating_Load", "Cooling_Load"])
y = df["Heating_Load"]
# Split into train/test sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=42)

# Normalize features

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 4. Train Linear Regression Model

model = LinearRegression()
model.fit(X_train, y_train)

# Step 5. Evaluate Performance

y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("RMSE:", rmse)
print("MAE:", mae)
print("R² Score:", r2)

'''Step 6. Optimization (Bonus Challenges)
Add polynomial features to capture non-linear relationships
(PolynomialFeatures in sklearn).
- Try regularization techniques (Ridge, Lasso).
- Perform feature importance analysis (coefficients).
- Apply cross-validation to ensure stability.

🔧 Your Task
- Choose one dataset (UCI or Kaggle).
- Load, clean, and preprocess the data.
- Train a linear regression model to predict the selected target variable.
- Evaluate the model using RMSE, MAE, and R².
- Improve the model with polynomial features or regularization. '''
