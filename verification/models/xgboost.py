import numpy as np
import pandas as pd
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import os


def xgboost_model(
    x, y, df, pred_days, exercised, burned, possible_cheat_days, fasting, classes, week
):
    # XGBoost model

    xgb_predictions = []

    last_actual_weight = df[df["day"] == 41]["weight"].values[0]
    lag1 = last_actual_weight

    model_xgb = XGBRegressor(
        n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42
    )
    model_xgb.fit(x, y)

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
        pred_weight = model_xgb.predict(features)[0]
        xgb_predictions.append(pred_weight)
        lag1 = pred_weight

    return xgb_predictions
