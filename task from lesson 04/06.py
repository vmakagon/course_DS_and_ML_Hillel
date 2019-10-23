# 6) * Построить регрессию полиномом 3й степени a*x3 + b*x2 + c*x + d,
# результат отобразить, и предсказать цену продажи квартиры 71m2.

import numpy as np
import matplotlib.pyplot as plt

def normalization(X):
    mean = X.mean()
    std = X.std()
    normal = (X - mean)/std
    return normal,mean,std

def plot_result3(X,Y, poly3model=None, extra_point=None):
    plt.xlabel('size,m2')
    plt.ylabel('price,$')
    plt.plot(X, Y, 'bo') # b-blue, o-dot, ro, gx
    if poly3model:
        b = poly3model[0]
        k = poly3model[1]
        k2 = poly3model[2]
        k3 = poly3model[3]
        t = np.arange(X.min(), X.max(), 0.01)
        plt.plot(t, k3*t**3 + k2*t**2 + k*t + b, 'k')
        if extra_point:
            y = k3*extra_point**3 + k2*extra_point**2 + k*extra_point + b
            plt.plot(extra_point, y, 'ro')
    plt.grid()
    plt.show()

def error_function3(X,Y,poly3model):
    b = poly3model[0]
    k = poly3model[1]
    k2 = poly3model[2]
    k3 = poly3model[3]
    J = (1/len(X)) * np.sum(((k3*X**3 + k2*X**2 + k*X + b) - Y)**2)
    return J

def gradient_desceent3(X,Y, alpha, n_steps, poly3model, is_printed=False):
    b = poly3model[0]
    k = poly3model[1]
    k2 = poly3model[2]
    k3 = poly3model[3]
    J_pred = 0
    for s in range(n_steps):
        db  = (1 / len(X)) * np.sum(( k3*X**3 + k2*X**2 + k*X + b) - Y)
        dk  = (1 / len(X)) * np.sum(((k3*X**3 + k2*X**2 + k*X + b) - Y) * X)
        dk2 = (1 / len(X)) * np.sum(((k3*X**3 + k2*X**2 + k*X + b) - Y) * X**2)
        dk3 = (1 / len(X)) * np.sum(((k3*X**3 + k2*X**2 + k*X + b) - Y) * X**3)
        b  -= alpha * db
        k  -= alpha * dk
        k2 -= alpha * dk2
        k3 -= alpha * dk3
        J = error_function3(X,Y,[b,k,k2,k3])
        if is_printed or (s % 10000) == 0:
            print(s,' error_function3', J,' difference ',J-J_pred,'\nModel',b,k,k2,k3,'\n db, dk, dk2, dk3 ',db,dk,dk2,dk3,'\n')
            J_pred = J
    return[b,k,k2,k3]

# data_set = np.array(data_set)
data_set = np.array(  [[76, 106141],
            [71, 115000],
            [76, 115000],
            [76, 106141],
            [76, 106141],
            [91, 128159],
            [186, 220000],
            [96, 130000],
            [62, 90000],
            [96, 130000],
            [43, 72000],
            [141, 180000],
            [70, 115000],
            [91, 128159],
            [92, 135000],
            [181, 240000],
            [90, 132700],
            [62, 93000],
            [72, 125000],
            [92, 145000],
            [186, 249000],
            [93, 140000],
            [91, 150000],
            [48, 67000],
            [91, 150000],
            [92, 140000],
            [76, 106141],
            [49, 87000],
            [93, 140000],
            [43, 63000]])

print(data_set)
X = data_set[:,0]
Y = data_set[:,1]
X, x_mean, x_std = normalization(X)
Y, y_mean, y_std = normalization(Y)
plot_result3(X,Y)

start_poly3model = [0.1,1,0.01,0.01]
plot_result3(X,Y,start_poly3model)
print(error_function3(X,Y,start_poly3model))
poly3model = gradient_desceent3(X,Y, 0.0001, 100000, start_poly3model)
print(poly3model)
print(error_function3(X,Y,poly3model))
predict_x = 71
x_temp = (predict_x - x_mean)/x_std
predict_y = (poly3model[3]*x_temp**3 + poly3model[2]*x_temp**2 +
             poly3model[1]*x_temp + poly3model[0])*y_std + y_mean
print("Прогноз цены для квартиры площадью {}м2 - {}$ (регрессия полиномом 3й степени)".format(predict_x, int(predict_y)))
plot_result3(X,Y,poly3model,x_temp)


