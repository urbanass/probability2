# Известно, что рост футболистов в сборной распределен нормально
# с дисперсией генеральной совокупности, равной 25 кв.см. Объем выборки равен 27,
# среднее выборочное составляет 174.2. Найдите доверительный интервал для
# математического ожидания с надежностью 0.95.

import numpy as np
from scipy import stats

var = 25
x = 27
mean = 174.2
std = (var)**0.5
a = 0.05
z = stats.norm.ppf(1-a/2, x-1)
d = z*std/(x)**0.5
min = mean - d
max = mean + d
print(f'>>> Доверительный интервал :{min: .3f} - {max: .3f}')