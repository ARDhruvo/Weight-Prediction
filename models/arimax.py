import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
import matplotlib.pyplot as plt
import os


def arimax_model(
    x, y, df, pred_days, exercised, burned, possible_cheat_days, fasting, classes, week
):
    # ARIMAX model

    arimax_predictions = []

    last_actual_weight = df[df["day"] == 41]["weight"].values[0]
    lag1 = last_actual_weight

    endog = df["weight"].values
    exog = df[
        ["day", "exercised", "cheat", "fasting", "week", "burned", "classes"]
    ].values

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
        cheat = 1 if (day in possible_cheat_days) else 0
        features = np.array([[day, exercised, cheat, fasting, week, burned, classes]])
        pred_weight = moodel_arimax.predict(n_periods=1, exogenous=features)[0]
        arimax_predictions.append(pred_weight)
        week += 1

    return arimax_predictions
