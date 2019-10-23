# 2. Дан список (list) цен на iphone xs max 256gb у разных продавцов на hotline:
#
# [47.999, 42.999, 49.999, 37.245, 38.324, 37.166, 38.988, 37.720]
#
# Средствами python, написать функцию, возвращающую tuple из min,
# max и mean (среднюю) и median (медианную) цену.
import numpy as np
a = [47.999, 42.999, 49.999, 37.245, 38.324, 37.166, 38.988, 37.720]
t = tuple()
b = [min(a), max(a), np.mean(a), np.median(a)]
t = tuple(b)
print(t,type(t))
