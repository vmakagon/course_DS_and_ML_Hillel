import numpy as np
import matplotlib.pyplot as plt

def star_plt():
    fig = plt.figure(figsize=(6, 6))
    plt.xlim(-2, 4)
    plt.ylim(-0.5, 4)

def end_plt():
    plt.grid()
    plt.show()

def plt_figure(X):
    plt.scatter(X[:, 0], X[:, 1], s=200, color=color)
    pl = plt.Polygon(X[:3], edgecolor='blue',facecolor='none')
    plt.gca().add_patch(pl)

def plt_figure_different(X0,X,title=''):
    star_plt()
    plt.title(title)
    plt_figure(X0)
    plt_figure(X)
    end_plt()

color = ['red','green','yellow']
X = np.array(  [[0., 0.],
                [2., 3.],
                [2.5, 1.5]])
angle = np.pi/6
M_rotate = np.array([[np.cos(angle), np.sin(angle)],
                     [-np.sin(angle), np.cos(angle)]])
M_scale = np.array([[0.333, 0.],
                    [0., 1.]])
plt_figure_different(X,X.dot(M_rotate),'1 Rotate ')
plt_figure_different(X.dot(M_rotate),X.dot(M_rotate).dot(M_scale),'2 Scale ')
plt_figure_different(X,X.dot(M_scale),'1 Scale ')
plt_figure_different(X.dot(M_scale),X.dot(M_scale).dot(M_rotate),'2 Rotate ')
M_result1 = M_rotate.dot(M_scale)
print(M_result1)
M_result2 = M_scale.dot(M_rotate)
print(M_result2)
print('Порядок умножения матриц важен: A*B != B*A')


