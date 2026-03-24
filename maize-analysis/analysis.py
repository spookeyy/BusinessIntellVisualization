import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# loading dataset
try:
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "maize_dataset.csv")
    df = pd.read_csv(csv_path)
except FileNotFoundError:
    print("ERROR: maize_dataset.csv not found in folder")
    exit()

print("\nDataset Loaded Successfully")
print(df.head())

# data cleaning
print("\nChecking for missing values...")
print(df.isnull().sum())

# Fill missing values
df.ffill(inplace=True)

numeric_cols = ["Rainfall_mm", "Fertilizer_MT", "Soil_Index", "Maize_Production_Bags"]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df.dropna(inplace=True)

# Convert bags to millions
df["Production_Million"] = df["Maize_Production_Bags"] / 1_000_000

# visualize

# Trend line
plt.figure()
plt.plot(df["Year"], df["Production_Million"], marker='o')
plt.title("Maize Production Trend (Million Bags)")
plt.xlabel("Year")
plt.ylabel("Production (Million Bags)")
plt.grid()
plt.show()

# Rainfall vs Yield
plt.figure()
plt.scatter(df["Rainfall_mm"], df["Production_Million"])
plt.title("Rainfall vs Maize Production")
plt.xlabel("Rainfall (mm)")
plt.ylabel("Production (Million Bags)")
plt.show()

# ML model
X = df[["Rainfall_mm", "Fertilizer_MT", "Soil_Index"]]
y = df["Production_Million"]

if len(df) < 10:
    test_size = 0.3
else:
    test_size = 0.2

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42
)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print("MAE:", round(mae, 3))
print("R2 Score:", round(r2, 3))

importance = model.feature_importances_

plt.figure()
plt.bar(["Rainfall", "Fertilizer", "Soil"], importance)
plt.title("Feature Importance")
plt.show()

# Predictions
plt.figure()
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Production")
plt.ylabel("Predicted Production")
plt.title("Actual vs Predicted")
plt.grid()
plt.show()

print("\nAnalysis Complete Successfully")
