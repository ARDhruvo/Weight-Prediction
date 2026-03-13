import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import os

output_dir = "predictions"
os.makedirs(output_dir, exist_ok=True)  # create if it doesn't exist


input_file = os.path.join(".", "data.csv")  # current directory / data.csv
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

features = ["day", "burned", "cheat", "fasting", "lag1"]

df["target"] = df["weight"].shift(-1)
df = df.dropna()
df_train = df.copy()

x = df[features]
y = df["target"]

fasting_days = list(range(12, 42))
possible_cheat_days = [42, 43, 44, 45]

# Linear regression model

linear_predictions = []

original_df = pd.read_csv(input_file)

last_actual_weight = df[df["day"] == 32]["weight"].values[0]
lag1 = last_actual_weight

pred_days = list(range(33, 57))

model_lr = LinearRegression()
model_lr.fit(x, y)

for i, day in enumerate(pred_days):
    burned = 0
    cheat = 1 if (day in possible_cheat_days) else 0
    fasting = 1 if (day in fasting_days) else 0
    features = np.array([[day, burned, cheat, fasting, lag1]])
    pred_weight = model_lr.predict(features)[0]
    linear_predictions.append(pred_weight)
    lag1 = pred_weight

print("Linear Regression Predictions:")
for i, pred in enumerate(linear_predictions):
    print(f"Day {pred_days[i]}: {pred:.2f}")

# ARIMA model

arima_whitenoise_predictions = []
arima_randomwalk_predictions = []
arima_autoregressive_predictions = []
arima_suggested_predictions = []

arima_whitenoise_model = ARIMA(df["weight"], order=(0, 0, 0))
arima_randomwalk_model = ARIMA(df["weight"], order=(0, 1, 0))
arima_autoregressive_model = ARIMA(df["weight"], order=(1, 0, 0))
arima_suggested_model = ARIMA(df["weight"], order=(5, 1, 0))

arima_whitenoise_fit = arima_whitenoise_model.fit()
arima_randomwalk_fit = arima_randomwalk_model.fit()
arima_autoregressive_fit = arima_autoregressive_model.fit()
arima_suggested_fit = arima_suggested_model.fit()

arima_whitenoise_forecast = arima_whitenoise_fit.get_forecast(steps=28)
arima_randomwalk_forecast = arima_randomwalk_fit.get_forecast(steps=28)
arima_autoregressive_forecast = arima_autoregressive_fit.get_forecast(steps=28)
arima_suggested_forecast = arima_suggested_fit.get_forecast(steps=28)

for d, pred in zip(pred_days, arima_whitenoise_forecast.predicted_mean):
    arima_whitenoise_predictions.append(pred)

for d, pred in zip(pred_days, arima_randomwalk_forecast.predicted_mean):
    arima_randomwalk_predictions.append(pred)

for d, pred in zip(pred_days, arima_autoregressive_forecast.predicted_mean):
    arima_autoregressive_predictions.append(pred)

for d, pred in zip(pred_days, arima_suggested_forecast.predicted_mean):
    arima_suggested_predictions.append(pred)

print("\nARIMA Predictions:")

print("White Noise Model:")
for i, pred in enumerate(arima_whitenoise_predictions):
    print(f"Day {pred_days[i]}: {pred:.2f}")

print("\nRandom Walk Model:")
for i, pred in enumerate(arima_randomwalk_predictions):
    print(f"Day {pred_days[i]}: {pred:.2f}")

print("\nAutoregressive Model:")
for i, pred in enumerate(arima_autoregressive_predictions):
    print(f"Day {pred_days[i]}: {pred:.2f}")

print("\nSuggested ARIMA Model:")
for i, pred in enumerate(arima_suggested_predictions):
    print(f"Day {pred_days[i]}: {pred:.2f}")


results_df = pd.DataFrame(
    {
        "day": pred_days,
        "linear_regression": linear_predictions,
        "arima_whitenoise": arima_whitenoise_predictions,
        "arima_randomwalk": arima_randomwalk_predictions,
        "arima_autoregressive": arima_autoregressive_predictions,
        "arima_suggested": arima_suggested_predictions,
    }
)

csv_path = os.path.join(output_dir, "predictions.csv")
results_df.to_csv(csv_path, index=False)
print(f"All predictions saved to {csv_path}")

plt.figure(figsize=(10, 6))
plt.plot(
    original_df["day"], original_df["weight"], "o-", label="Historical", color="black"
)
plt.plot(pred_days, linear_predictions, "x--", label="Linear Regression", color="red")
plt.plot(
    pred_days, arima_whitenoise_predictions, label="ARIMA White Noise", color="blue"
)
plt.plot(
    pred_days, arima_randomwalk_predictions, label="ARIMA Random Walk", color="green"
)
plt.plot(
    pred_days, arima_autoregressive_predictions, label="ARIMA AR(1)", color="orange"
)
plt.plot(pred_days, arima_suggested_predictions, label="ARIMA (5,1,0)", color="purple")
plt.xlabel("Day")
plt.ylabel("Weight")
plt.title("Weight Forecasts for Next 28 Days")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, "forecast_comparison.png"))
plt.close()
