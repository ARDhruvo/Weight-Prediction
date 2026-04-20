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
        cheat = 1 if (day in possible_cheat_days) else 0
        features = np.array(
            [[day, exercised, cheat, fasting, week, burned, classes, lag1]]
        )
        pred_weight = model_ridge.predict(features)[0]
        ridge_predictions.append(pred_weight)
        lag1 = pred_weight
        week += 1

    return ridge_predictions
