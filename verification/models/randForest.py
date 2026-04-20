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
        cheat_pred = possible_cheat_days[i]
        day_pred = pred_days[i]
        exercised_pred = exercised[i]
        fasting_pred = fasting[i]
        burned_pred = burned[i]
        classes_pred = classes[i]
        week_pred = week[i]

        features = np.array(
            [
                [
                    day_pred,
                    exercised_pred,
                    cheat_pred,
                    fasting_pred,
                    week_pred,
                    burned_pred,
                    classes_pred,
                    lag1,
                ]
            ]
        )
        pred_weight = model_rf.predict(features)[0]
        randForest_predictions.append(pred_weight)
        lag1 = pred_weight

    return randForest_predictions
