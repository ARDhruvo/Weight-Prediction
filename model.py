import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Future idea: Make it modular

script_dir = os.path.dirname(__file__)
input_file = os.path.join(script_dir, "data.csv")
output_dir = os.path.join(script_dir, "predictions")
os.makedirs(output_dir, exist_ok=True)
df = pd.read_csv(input_file)

# Graphing the data

x = df["day"].values.reshape(-1, 1)
y = df["weight"].values

plt.scatter(x, y, color="blue")
plt.xlabel("Day")
plt.ylabel("Weight")
plt.title("Weight vs Day")
plt.savefig(os.path.join(output_dir, "weight_plot.png"))

# Training the model

df["lag1"] = df["weight"].shift(1)

features = ["lag1", "day", "exercised", "cheat", "fasting", "week", "burned", "classes"]

df["target"] = df["weight"].shift(-1)
df = df.dropna()
df_train = df.copy()

x = df[features]
y = df["target"]


# Extra stuffs

pred_days = list(range(43, 50))

possible_cheat_days = [42, 43, 44]

week = 0

original_df = pd.read_csv(input_file)


# Linear regression model

linear_predictions = []

last_actual_weight = df[df["day"] == 41]["weight"].values[0]
lag1 = last_actual_weight


model_lr = LinearRegression()
model_lr.fit(x, y)


for i, day in enumerate(pred_days):
    exercised = 0
    burned = 80
    cheat = 1 if (day in possible_cheat_days) else 0
    fasting = 0
    classes = 0
    features = np.array([[day, exercised, cheat, fasting, week, burned, classes, lag1]])
    pred_weight = model_lr.predict(features)[0]
    linear_predictions.append(pred_weight)
    lag1 = pred_weight
    week += 1

week = 0

print("Linear Regression Predictions:")
for i, pred in enumerate(linear_predictions):
    print(f"Day {pred_days[i]}: {pred:.2f}")


# Random Forest model

randForest_predictions = []

last_actual_weight = df[df["day"] == 41]["weight"].values[0]
lag1 = last_actual_weight

model_rf = RandomForestRegressor(
    n_estimators=100, max_depth=20, min_samples_leaf=5, random_state=42
)
model_rf.fit(x, y)

for i, day in enumerate(pred_days):
    exercised = 0
    burned = 80
    cheat = 1 if (day in possible_cheat_days) else 0
    fasting = 0
    classes = 0
    features = np.array([[day, exercised, cheat, fasting, week, burned, classes, lag1]])
    pred_weight = model_rf.predict(features)[0]
    randForest_predictions.append(pred_weight)
    lag1 = pred_weight
    week += 1

week = 0

print("Random Forest Predictions:")
for i, pred in enumerate(randForest_predictions):
    print(f"Day {pred_days[i]}: {pred:.2f}")


# Ridge Regression model

ridge_predictions = []

last_actual_weight = df[df["day"] == 41]["weight"].values[0]
lag1 = last_actual_weight

model_ridge = Ridge(alpha=1.0)
model_ridge.fit(x, y)

for i, day in enumerate(pred_days):
    exercised = 0
    burned = 80
    cheat = 1 if (day in possible_cheat_days) else 0
    fasting = 0
    classes = 0
    features = np.array([[day, exercised, cheat, fasting, week, burned, classes, lag1]])
    pred_weight = model_ridge.predict(features)[0]
    ridge_predictions.append(pred_weight)
    lag1 = pred_weight
    week += 1

week = 0

print("Ridge Regression Predictions:")
for i, pred in enumerate(ridge_predictions):
    print(f"Day {pred_days[i]}: {pred:.2f}")

# ARIMAX model

arimax_predictions = []

last_actual_weight = df[df["day"] == 41]["weight"].values[0]
lag1 = last_actual_weight

endog = df["weight"].values
exog = df[["day", "exercised", "cheat", "fasting", "week", "burned", "classes"]].values

moodel_arimax = auto_arima(
    endog,
    exogenous=exog,
    start_p=0,
    max_p=3,
    start_q=0,
    max_q=3,
    d=None,
    max_d=2,
    seasonal=False,
    stepwise=True,
    trace=True,
    suppress_warnings=True,
    error_action="ignore",
)

for i, day in enumerate(pred_days):
    exercised = 0
    burned = 80
    cheat = 1 if (day in possible_cheat_days) else 0
    fasting = 0
    classes = 0
    features = np.array([[day, exercised, cheat, fasting, week, burned, classes]])
    pred_weight = moodel_arimax.predict(n_periods=1, exogenous=features)[0]
    arimax_predictions.append(pred_weight)
    week += 1

week = 0

print("ARIMAX Predictions:")
for i, pred in enumerate(arimax_predictions):
    print(f"Day {pred_days[i]}: {pred:.2f}")

# XGBoost model

xgb_predictions = []

last_actual_weight = df[df["day"] == 41]["weight"].values[0]
lag1 = last_actual_weight

model_xgb = XGBRegressor(
    n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42
)
model_xgb.fit(x, y)
for i, day in enumerate(pred_days):
    exercised = 0
    burned = 80
    cheat = 1 if (day in possible_cheat_days) else 0
    fasting = 0
    classes = 0
    features = np.array([[day, exercised, cheat, fasting, week, burned, classes, lag1]])
    pred_weight = model_xgb.predict(features)[0]
    xgb_predictions.append(pred_weight)
    lag1 = pred_weight
    week += 1

week = 0

print("XGBoost Predictions:")
for i, pred in enumerate(xgb_predictions):
    print(f"Day {pred_days[i]}: {pred:.2f}")

# Saving Results

results_df = pd.DataFrame({"day": pred_days, "linear_regression": linear_predictions})
results_df["ridge_regression"] = ridge_predictions
results_df["random_forest"] = randForest_predictions
results_df["arimax"] = arimax_predictions
results_df["xgboost"] = xgb_predictions

csv_path = os.path.join(output_dir, "predictions.csv")
results_df.to_csv(csv_path, index=False)
print(f"All predictions saved to {csv_path}")

plt.figure(figsize=(10, 6))
plt.plot(
    original_df["day"], original_df["weight"], "o-", label="Historical", color="black"
)
plt.plot(pred_days, linear_predictions, "x--", label="Linear Regression", color="red")
plt.plot(pred_days, randForest_predictions, "s--", label="Random Forest", color="green")
plt.plot(pred_days, ridge_predictions, "o--", label="Ridge Regression", color="purple")
plt.plot(pred_days, arimax_predictions, "d--", label="ARIMAX", color="blue")
plt.plot(pred_days, xgb_predictions, "p--", label="XGBoost", color="orange")
plt.xlabel("Day")
plt.ylabel("Weight")
plt.title("Weight Forecasts for Next 28 Days")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, "forecast_comparison.png"))
plt.close()
