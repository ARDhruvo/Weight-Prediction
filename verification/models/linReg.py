import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import os


def linear_model(
    x, y, df, pred_days, exercised, burned, possible_cheat_days, fasting, classes, week
):
    # Linear regression model

    linear_predictions = []

    last_actual_weight = df[df["day"] == 41]["weight"].values[0]
    lag1 = last_actual_weight

    model_lr = LinearRegression()
    model_lr.fit(x, y)

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
        pred_weight = model_lr.predict(features)[0]
        linear_predictions.append(pred_weight)
        lag1 = pred_weight

    return linear_predictions
