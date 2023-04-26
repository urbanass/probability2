# Произвести вычисления как в пункте 2, но с вычислением intercept. Учесть, что
# изменение коэффициентов должно производиться
# на каждом шаге одновременно (то есть изменение одного коэффициента не должно
# влиять на изменение другого во время одной итерации).

import numpy as np
import matplotlib.pyplot as plt

alpha = 5e-5

zp = np.array([35, 45, 190, 200, 40, 70, 54, 150, 120, 110])
ks = np.array([401, 574, 874, 919, 459, 739, 653, 902, 746, 832])

n = len(ks)
b1 = (np.mean(zp*ks)-np.mean(zp)*np.mean(ks))/(np.mean(zp**2) - np.mean(zp)**2)
b0 = np.mean(ks)-b1*np.mean(zp)
y_pred = b0 + b1 * zp
plt.scatter(zp, ks)
plt.plot(zp, y_pred)
mse_ = np.sum(((b0 + b1 * zp) - ks) ** 2 / n)
alpha = 1e-6
b0 = 0.1
b1 = 0.1

def mse_(b0, b1, y=ks, X=zp, n=10):
    return np.sum((b0 + b1 * X - y) ** 2) / n


for i in range(1000000):
    y_pred2 = b0 + b1 * zp
    b0 -= alpha * (2 / n) * np.sum((y_pred2 - ks))
    b1 -= alpha * (2 / n) * np.sum((y_pred2 - ks)*zp)
    if i % 100000 == 0:
        print(f"{i}, b1 : {b1}, b0 : {b0}, mse: {mse_(b0,b1)}")
