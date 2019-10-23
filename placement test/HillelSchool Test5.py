# 5. Нарисуйте трехмерные оси координат.
# Постройте на них вектора i = (2, 0, 0), j = (0, 3, 0), k = (0, 0, 5).
# Постройте вектор, являющийся их суммой: b = i + j + k

import matplotlib as mpl
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
i = np.array([2,0,0])
j = np.array([0,3,0])
k = np.array([0,0,5])
b = i + j + k
print(b)
data=np.array([i,j,k,b])
fig=plt.figure()
ax=Axes3D(fig)
ax.set_xlim3d(0,6)
ax.set_ylim3d(0,6)
ax.set_zlim3d(0,6)
ax.scatter(data[:,0],data[:,1],data[:,2])
ax.quiver(0,0,0,i[0],i[1],i[2])
ax.quiver(0,0,0,j[0],j[1],j[2])
ax.quiver(0,0,0,k[0],k[1],k[2])
ax.quiver(0,0,0,b[0],b[1],b[2])
plt.show()