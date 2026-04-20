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

    n_forecast = len(pred_days)

    for i in range(n_forecast):
        exog_future = np.array(
            [
                pred_days[i],
                exercised[i],
                possible_cheat_days[i],
                fasting[i],
                week[i],
                burned[i],
                classes[i],
            ]
        ).reshape(1, -1)

        pred = moodel_arimax.predict(n_periods=1, exogenous=exog_future)[0]
        arimax_predictions.append(pred)

    return arimax_predictions
