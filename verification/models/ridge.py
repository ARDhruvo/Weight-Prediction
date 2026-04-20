import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
import os


def ridge_model(
    x, y, df, pred_days, exercised, burned, possible_cheat_days, fasting, classes, week
):
    # Ridge Regression model

    ridge_predictions = []

    last_actual_weight = df[df["day"] == 41]["weight"].values[0]
    lag1 = last_actual_weight

    model_ridge = Ridge(alpha=1.0)
    model_ridge.fit(x, y)

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
        pred_weight = model_ridge.predict(features)[0]
        ridge_predictions.append(pred_weight)
        lag1 = pred_weight

    return ridge_predictions
