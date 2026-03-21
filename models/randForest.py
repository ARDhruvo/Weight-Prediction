import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import os


def random_forest_model(
    x, y, df, pred_days, exercised, burned, possible_cheat_days, fasting, classes, week
):
    # Random Forest model

    randForest_predictions = []

    last_actual_weight = df[df["day"] == 41]["weight"].values[0]
    lag1 = last_actual_weight

    model_rf = RandomForestRegressor(
        n_estimators=100, max_depth=20, min_samples_leaf=5, random_state=42
    )
    model_rf.fit(x, y)

    for i, day in enumerate(pred_days):
        cheat = 1 if (day in possible_cheat_days) else 0

        features = np.array(
            [[day, exercised, cheat, fasting, week, burned, classes, lag1]]
        )
        pred_weight = model_rf.predict(features)[0]
        randForest_predictions.append(pred_weight)
        lag1 = pred_weight
        week += 1

    return randForest_predictions
