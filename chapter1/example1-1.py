import pandas as pd
from chapter1.LinearRegression import LinearRegression
import numpy as np

data_root = "https://github.com/ageron/data/raw/main/"
life_sat = pd.read_csv(data_root + "lifesat/lifesat.csv")

X = life_sat.drop(["Life satisfaction", "Country"], axis=1).values
y = life_sat["Life satisfaction"].values

# нормализация
X_mean = X.mean()
X_std = X.std()
X_norm = (X - X_mean) / X_std

model = LinearRegression(lr=0.01, iterations=1000)
model.fit(X_norm, y)

# предсказание — тоже нормализуй вход
data = 33_442.8
data_norm = (np.array([[data]]) - X_mean) / X_std
prediction = model.predict(data_norm)
print(prediction)