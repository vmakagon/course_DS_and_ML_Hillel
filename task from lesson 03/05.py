import numpy as np
X = np.array([[0. , 0. ],
              [2. , 3. ],
              [2.5, 1.5]])
Y = np.array([[ 0.     ,  0.     ],
              [-0.3535 , -2.4745 ],
              [-1.23725, -2.29775]])
v1 = X[1]
v2 = X[2]
V = np.array([[v1[0],v2[0]],[v1[1],v2[1]]])
w1 = X[1]
w2 = X[2]
W = np.array([[w1[0],w2[0]],[w1[1],w2[1]]])
print('V:',V)
print('W',W)
print('A*V = W')
print('Домножаем на Vобратную \nA*V*Vобр = W*Vобр')
print('A*I = A = W*Vобр , где I- единичная матрица')
A = W.dot(np.linalg.inv(V))
print(A)
