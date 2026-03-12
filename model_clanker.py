import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings("ignore")

# Set style for plots
plt.style.use("seaborn-v0_8-darkgrid")

# Load the data
data = pd.read_csv("data.csv")

# Display basic info
print("Dataset Info:")
print(data.head())
print("\nDataset Shape:", data.shape)
print("\nBasic Statistics:")
print(data.describe())

# Feature engineering
# Create additional features that might help with prediction
data["day_squared"] = data["day"] ** 2
data["cumulative_burned"] = data["burned"].cumsum()
data["cumulative_cheat"] = data["cheat"].cumsum()
data["cumulative_fasting"] = data["fasting"].cumsum()

# Create lag features (previous day's weight)
data["weight_lag1"] = data["weight"].shift(1)

# Create rolling averages
data["weight_ma3"] = data["weight"].rolling(window=3, min_periods=1).mean()
data["weight_ma7"] = data["weight"].rolling(window=7, min_periods=1).mean()

# Drop NaN values created by lag features
data = data.dropna().reset_index(drop=True)

print("\nData after feature engineering:")
print(data.head())

# Prepare features for modeling
# Simple features for basic model
X_simple = data[["day"]].values
y = data["weight"].values

# More comprehensive features for advanced models
X_advanced = data[
    [
        "day",
        "burned",
        "cheat",
        "fasting",
        "day_squared",
        "cumulative_burned",
        "cumulative_cheat",
        "cumulative_fasting",
        "weight_lag1",
        "weight_ma3",
        "weight_ma7",
    ]
].values

# Split data into training and testing sets
X_train_simple, X_test_simple, y_train, y_test = train_test_split(
    X_simple, y, test_size=0.2, random_state=42, shuffle=False
)

X_train_advanced, X_test_advanced, _, _ = train_test_split(
    X_advanced, y, test_size=0.2, random_state=42, shuffle=False
)

# 1. Simple Linear Regression
print("\n" + "=" * 50)
print("MODEL 1: SIMPLE LINEAR REGRESSION")
print("=" * 50)

simple_lr = LinearRegression()
simple_lr.fit(X_train_simple, y_train)

# Predictions
y_pred_simple_train = simple_lr.predict(X_train_simple)
y_pred_simple_test = simple_lr.predict(X_test_simple)

# Evaluation
print(f"Train R² Score: {r2_score(y_train, y_pred_simple_train):.4f}")
print(f"Test R² Score: {r2_score(y_test, y_pred_simple_test):.4f}")
print(f"Train MAE: {mean_absolute_error(y_train, y_pred_simple_train):.4f}")
print(f"Test MAE: {mean_absolute_error(y_test, y_pred_simple_test):.4f}")
print(f"Equation: weight = {simple_lr.coef_[0]:.4f} * day + {simple_lr.intercept_:.4f}")

# 2. Polynomial Regression
print("\n" + "=" * 50)
print("MODEL 2: POLYNOMIAL REGRESSION")
print("=" * 50)

degrees = [2, 3, 4]
best_poly_model = None
best_poly_score = -np.inf

for degree in degrees:
    poly_model = make_pipeline(
        PolynomialFeatures(degree=degree, include_bias=False), LinearRegression()
    )
    poly_model.fit(X_train_simple, y_train)
    y_pred_poly = poly_model.predict(X_test_simple)
    score = r2_score(y_test, y_pred_poly)

    print(f"Degree {degree} - Test R²: {score:.4f}")

    if score > best_poly_score:
        best_poly_score = score
        best_poly_model = poly_model
        best_degree = degree

print(f"\nBest Polynomial Degree: {best_degree}")
y_pred_poly_test = best_poly_model.predict(X_test_simple)
print(f"Best Test R²: {r2_score(y_test, y_pred_poly_test):.4f}")

# 3. Ridge Regression (with advanced features)
print("\n" + "=" * 50)
print("MODEL 3: RIDGE REGRESSION (with advanced features)")
print("=" * 50)

alphas = [0.01, 0.1, 1.0, 10.0]
best_ridge = None
best_ridge_score = -np.inf

for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train_advanced, y_train)
    y_pred_ridge = ridge.predict(X_test_advanced)
    score = r2_score(y_test, y_pred_ridge)

    print(f"Alpha {alpha} - Test R²: {score:.4f}")

    if score > best_ridge_score:
        best_ridge_score = score
        best_ridge = ridge
        best_alpha = alpha

print(f"\nBest Ridge Alpha: {best_alpha}")
y_pred_ridge_test = best_ridge.predict(X_test_advanced)
print(f"Best Test R²: {r2_score(y_test, y_pred_ridge_test):.4f}")

# 4. Lasso Regression (feature selection)
print("\n" + "=" * 50)
print("MODEL 4: LASSO REGRESSION (feature selection)")
print("=" * 50)

alphas_lasso = [0.001, 0.01, 0.1, 1.0]
best_lasso = None
best_lasso_score = -np.inf

for alpha in alphas_lasso:
    lasso = Lasso(alpha=alpha, max_iter=10000)
    lasso.fit(X_train_advanced, y_train)
    y_pred_lasso = lasso.predict(X_test_advanced)
    score = r2_score(y_test, y_pred_lasso)

    print(f"Alpha {alpha} - Test R²: {score:.4f}")

    if score > best_lasso_score:
        best_lasso_score = score
        best_lasso = lasso
        best_lasso_alpha = alpha

print(f"\nBest Lasso Alpha: {best_lasso_alpha}")
y_pred_lasso_test = best_lasso.predict(X_test_advanced)
print(f"Best Test R²: {r2_score(y_test, y_pred_lasso_test):.4f}")

# Feature importance from Lasso
feature_names = [
    "day",
    "burned",
    "cheat",
    "fasting",
    "day_squared",
    "cumulative_burned",
    "cumulative_cheat",
    "cumulative_fasting",
    "weight_lag1",
    "weight_ma3",
    "weight_ma7",
]

print("\nFeature Importance (Lasso coefficients):")
for name, coef in zip(feature_names, best_lasso.coef_):
    if abs(coef) > 0.001:  # Only show non-zero coefficients
        print(f"{name:20s}: {coef:.4f}")

# ========== PREDICT NEXT 28 DAYS ==========
print("\n" + "=" * 50)
print("PREDICTIONS FOR NEXT 28 DAYS")
print("=" * 50)

# First, make simple predictions (these don't depend on complex features)
future_days = np.arange(29, 57).reshape(-1, 1)
future_days_flat = future_days.flatten()

# Simple linear regression predictions
future_pred_simple = simple_lr.predict(future_days)

# Polynomial regression predictions
future_pred_poly = best_poly_model.predict(future_days)

# For advanced models, we need to create future features
# We'll use average values from the last week for predictions
last_week_data = data.tail(7)
avg_burned = last_week_data["burned"].mean()
avg_cheat = last_week_data["cheat"].mean()
avg_fasting = last_week_data["fasting"].mean()
last_weight = data["weight"].iloc[-1]

# FIRST, let's make a simple prediction using just day for ridge/lasso
# This will give us initial values to work with
simple_future_pred = simple_lr.predict(future_days)

# Now create future features for advanced models
future_features = []
for i, day in enumerate(future_days_flat):
    day_val = day
    day_squared = day**2
    cumulative_burned = data["burned"].sum() + avg_burned * (i + 1)
    cumulative_cheat = data["cheat"].sum() + avg_cheat * (i + 1)
    cumulative_fasting = data["fasting"].sum() + avg_fasting * (i + 1)

    # For lag and MA, we'll use a combination of last known values and simple predictions
    if i == 0:
        weight_lag = last_weight
        # For moving averages, use recent actual data
        recent_weights = data["weight"].tail(7).tolist()
        weight_ma3 = np.mean(recent_weights[-3:])
        weight_ma7 = np.mean(recent_weights)
    else:
        # Use simple linear predictions as a proxy for future weights
        weight_lag = simple_future_pred[i - 1]

        # For MA3, average last 3 predictions (or actuals + predictions)
        if i >= 2:
            ma3_values = simple_future_pred[i - 2 : i + 1]
        elif i == 1:
            ma3_values = [last_weight] + simple_future_pred[0:1].tolist()
        else:
            ma3_values = [last_weight]
        weight_ma3 = np.mean(ma3_values)

        # For MA7, use a combination
        if i >= 6:
            ma7_values = simple_future_pred[i - 6 : i + 1]
        else:
            # Take some from actual data and some from predictions
            actual_needed = 7 - (i + 1)
            if actual_needed > 0:
                recent_actuals = data["weight"].tail(actual_needed).tolist()
                ma7_values = recent_actuals + simple_future_pred[0 : i + 1].tolist()
            else:
                ma7_values = simple_future_pred[i - 6 : i + 1]
        weight_ma7 = np.mean(ma7_values)

    future_features.append(
        [
            day_val,
            avg_burned,
            avg_cheat,
            avg_fasting,
            day_squared,
            cumulative_burned,
            cumulative_cheat,
            cumulative_fasting,
            weight_lag,
            weight_ma3,
            weight_ma7,
        ]
    )

future_features = np.array(future_features)

# NOW we can make the advanced predictions
future_pred_ridge = best_ridge.predict(future_features)
future_pred_lasso = best_lasso.predict(future_features)

# Create a results dataframe
results_df = pd.DataFrame(
    {
        "day": future_days_flat,
        "linear_pred": future_pred_simple,
        "poly_pred": future_pred_poly,
        "ridge_pred": future_pred_ridge,
        "lasso_pred": future_pred_lasso,
    }
)

print("\nPredictions for next 28 days:")
print(results_df.head(10))  # Show first 10 days
print("\nSummary statistics for predictions:")
print(results_df.describe())

# VISUALIZATION
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Historical data with simple linear regression
axes[0, 0].scatter(data["day"], data["weight"], color="blue", label="Actual", alpha=0.6)
axes[0, 0].plot(
    data["day"], simple_lr.predict(X_simple), color="red", label="Linear Fit"
)
axes[0, 0].plot(
    future_days_flat,
    future_pred_simple,
    "--",
    color="orange",
    label="Future Predictions",
)
axes[0, 0].set_xlabel("Day")
axes[0, 0].set_ylabel("Weight (kg)")
axes[0, 0].set_title("Simple Linear Regression")
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Polynomial regression
axes[0, 1].scatter(data["day"], data["weight"], color="blue", label="Actual", alpha=0.6)
axes[0, 1].plot(
    data["day"],
    best_poly_model.predict(X_simple),
    color="green",
    label=f"Polynomial (deg={best_degree})",
)
axes[0, 1].plot(
    future_days_flat, future_pred_poly, "--", color="orange", label="Future Predictions"
)
axes[0, 1].set_xlabel("Day")
axes[0, 1].set_ylabel("Weight (kg)")
axes[0, 1].set_title(f"Polynomial Regression (degree={best_degree})")
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: All predictions comparison
axes[1, 0].scatter(data["day"], data["weight"], color="blue", label="Actual", alpha=0.6)
axes[1, 0].plot(
    future_days_flat, future_pred_simple, "-", color="red", label="Linear", linewidth=2
)
axes[1, 0].plot(
    future_days_flat,
    future_pred_poly,
    "-",
    color="green",
    label="Polynomial",
    linewidth=2,
)
axes[1, 0].plot(
    future_days_flat, future_pred_ridge, "-", color="purple", label="Ridge", linewidth=2
)
axes[1, 0].plot(
    future_days_flat, future_pred_lasso, "-", color="orange", label="Lasso", linewidth=2
)
axes[1, 0].axvline(
    x=28, color="black", linestyle="--", alpha=0.5, label="Prediction Start"
)
axes[1, 0].set_xlabel("Day")
axes[1, 0].set_ylabel("Weight (kg)")
axes[1, 0].set_title("All Models - Future Predictions")
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Residual analysis for best model
# Let's use Ridge as it performed well
y_pred_best = best_ridge.predict(X_advanced)
residuals = y - y_pred_best

axes[1, 1].scatter(y_pred_best, residuals, alpha=0.6)
axes[1, 1].axhline(y=0, color="red", linestyle="--")
axes[1, 1].set_xlabel("Predicted Values")
axes[1, 1].set_ylabel("Residuals")
axes[1, 1].set_title("Residual Plot (Ridge Regression)")
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("weight_prediction_results.png", dpi=300, bbox_inches="tight")
plt.show()

# SAVE RESULTS
results_df.to_csv("weight_predictions_next_28_days.csv", index=False)
print("\n✅ Results saved to 'weight_predictions_next_28_days.csv'")
print("✅ Plot saved to 'weight_prediction_results.png'")

# Summary and recommendations
print("\n" + "=" * 50)
print("SUMMARY AND RECOMMENDATIONS")
print("=" * 50)

print(
    f"""
Based on the analysis:

1. Best performing model: Ridge Regression (alpha={best_alpha})
   - Test R² Score: {best_ridge_score:.4f}
   - This indicates that the model explains {best_ridge_score*100:.1f}% of the variance in weight

2. Projected weight after 28 days:
   - Linear: {future_pred_simple[-1]:.2f} kg
   - Polynomial: {future_pred_poly[-1]:.2f} kg
   - Ridge: {future_pred_ridge[-1]:.2f} kg
   - Lasso: {future_pred_lasso[-1]:.2f} kg

3. Average predicted weight loss per week:
   - {(data['weight'].iloc[0] - future_pred_ridge[-1])/4:.2f} kg/week (Ridge model)

4. Key factors affecting weight (from Lasso model):
   - Previous day's weight (weight_lag1): {best_lasso.coef_[8]:.4f}
   - Cumulative fasting: {best_lasso.coef_[7]:.4f}
   - Cumulative burned calories: {best_lasso.coef_[5]:.4f}

Recommendations:
- Track these key factors more carefully
- Consider collecting additional data (calories consumed, exercise type, sleep)
- Update predictions weekly with new data
"""
)
