import pandas as pd
import numpy as np
from numpy import random
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

data_set = []
y_list = []
for _ in range(50):
    x_val = random.randint(1, 10)
    mul = random.uniform(0.7, 1.2)
    y_val = x_val * mul

df = pd.DataFrame(data_set, columns=["x_val", "y_val"])
print(df)

x = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

L_reg = LinearRegression()
L_reg.fit(x, y)
y_pre = L_reg.predict(x)
