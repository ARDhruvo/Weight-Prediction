import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

from models.linReg import linear_model
from models.randForest import random_forest_model
from models.ridge import ridge_model
from models.arimax import arimax_model
from models.xgboost import xgboost_model


warnings.filterwarnings("ignore")
script_dir = os.path.dirname(__file__)
input_file = os.path.join(script_dir, "..", "data.csv")
pred_file = os.path.join(script_dir, "..", "pred.csv")
output_dir = os.path.join(script_dir, "predictions")
os.makedirs(output_dir, exist_ok=True)
df = pd.read_csv(input_file)
pred_df = pd.read_csv(pred_file)

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

pred_days = pred_df["day"].values

possible_cheat_days = pred_df["cheat"].values

week = pred_df["week"].values
exercised = pred_df["exercised"].values
burned = pred_df["burned"].values
fasting = pred_df["fasting"].values
classes = pred_df["classes"].values

original_df = pd.read_csv(input_file)


linear_predictions = linear_model(
    x, y, df, pred_days, exercised, burned, possible_cheat_days, fasting, classes, week
)

randForest_predictions = random_forest_model(
    x, y, df, pred_days, exercised, burned, possible_cheat_days, fasting, classes, week
)

ridge_predictions = ridge_model(
    x, y, df, pred_days, exercised, burned, possible_cheat_days, fasting, classes, week
)

arimax_predictions = arimax_model(
    x, y, df, pred_days, exercised, burned, possible_cheat_days, fasting, classes, week
)

xgb_predictions = xgboost_model(
    x, y, df, pred_days, exercised, burned, possible_cheat_days, fasting, classes, week
)

# Displaying and Saving Results

print("\nSummary of Predictions:")
for i, day in enumerate(pred_days):
    print(
        f"Day {day}: Linear={linear_predictions[i]:.2f}, Ridge={ridge_predictions[i]:.2f}, "
        f"Random Forest={randForest_predictions[i]:.2f}, ARIMAX={arimax_predictions[i]:.2f}, "
        f"XGBoost={xgb_predictions[i]:.2f}"
    )

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
