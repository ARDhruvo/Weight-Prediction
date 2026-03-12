import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

df = pd.read_csv("data.csv")

# Graphing the data

x = df["day"].values.reshape(-1, 1)
y = df["weight"].values

plt.scatter(x, y, color="blue")
plt.xlabel("Day")
plt.ylabel("Weight")
plt.title("Weight vs Day")
plt.savefig("weight_plot.png")

# Training the model

df["lag1"] = df["weight"].shift(1)

x = df["day", "day,burned", "cheat", "fasting", "lag1"].values.reshape(-1, 5)
y = df["weight"].values

model = LinearRegression()
model.fit(x, y)
