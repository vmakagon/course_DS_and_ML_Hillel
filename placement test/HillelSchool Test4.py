# 4. Вычислите произведение матрицы M на вектор a.
# M = |1, 2|   a = |1|
#     |3, 4|       |2|
import numpy as np
M = np.array([
    [1,2],
    [3,4]
])
a = np.array([
    [1],
    [2]
])
print('1*1 + 2*2     5')
print('3*1 + 4*2     11')
print(M.dot(a))