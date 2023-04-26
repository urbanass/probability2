# Посчитать коэффициент линейной регрессии при заработной плате (zp), используя
# градиентный спуск (без intercept).

import numpy as np
import matplotlib.pyplot as plt

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
b1 = 0.1


def mse_(b1, y=ks, X=zp, n=10):
    return np.sum((b1 * X - y) ** 2) / n


for i in range(1000):
    fp = (1 / n) * np.sum(2 * (b1 * zp - ks) * zp)
    b1 -= alpha * fp
    if i % 100 == 0:
        print(f'Итерация: {i}, b1 : {b1}, mse: {mse_(b1) }')

y_pred2 = b1 * zp
print(y_pred2)

plt.scatter(zp, ks)
plt.plot(zp, y_pred, 'b:', label = 'с интерсептом')
plt.plot(zp, y_pred2, 'r:', label = 'без интерсепта')
plt.legend()