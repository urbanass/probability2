# Провести дисперсионный анализ для определения того, есть ли различия среднего роста среди взрослых футболистов, хоккеистов и штангистов.
# Даны значения роста в трех группах случайно выбранных спортсменов:

# Футболисты: 173, 175, 180, 178, 177, 185, 183, 182.
# Хоккеисты: 177, 179, 180, 188, 177, 172, 171, 184, 180.
# Штангисты: 172, 173, 169, 177, 166, 180, 178, 177, 172, 166, 170.

import scipy.stats as stats
import numpy as np

fp = np.array([173, 175, 180, 178, 177, 185, 183, 182])
hp = np.array([177, 179, 180, 188, 177, 172, 171, 184, 180])
wl = np.array([172, 173, 169, 177, 166, 180, 178, 177, 172, 166, 170])

n1 = fp.shape[0]
n2 = hp.shape[0]
n3 = wl.shape[0]

fp_mean = fp.mean()
hp_mean = hp.mean()
wl_mean = wl.mean()

print(f"Средний рост футболистов = {fp_mean:.3f}")
print(f"Средний рост хоккеистов = {hp_mean:.3f}")
print(f"Средний рост штангистов = {wl_mean:.3f}")

y = np.concatenate([fp, hp, wl])
y_mean = y.mean()

print(f"Средний рост спортсменов = {y_mean:.3f}")

S_F = n1 * (fp_mean - y_mean) ** 2 + n2 * (hp_mean - y_mean) ** 2 + n3 * (wl_mean - y_mean) ** 2
S_res = ((fp - fp_mean) ** 2).sum() + ((hp - hp_mean) ** 2).sum() + ((wl - wl_mean) ** 2).sum()

S = ((y - y_mean) ** 2).sum()

print(f"Sf^2 = {S_F:.3f}")
print(f"Sres^2 = {S_res:.3f}")
print(f"{S:.3f} = {S_F + S_res:.3f}")

k = 3
n = n1 + n2 + n3
k1 = k - 1
k2 = n - k

sigma_F = S_F / k1
sigma_res = S_res / k2

print(f"Факторная дисперсия = {sigma_F:.3f}")
print(f"Остаточная дисперсия = {sigma_res:.3f}")

T = sigma_F / sigma_res

print(f"Значение статистики = {T:.1f}")

print(stats.f_oneway(fp, hp, wl))

a = 0.05

F_crit = stats.f.ppf(1 - a, k1, k2)

print(f"Критическое значение = {F_crit:.3f}")

