# 7) ** Поочередно убрать один пример из датасета и построить линейную и кубическую регрессию
# для каждого из оставшихся примеров, предсказывая для каждого типа регрессий цену продажи квартиры 71m2.
# Вы должны получить m-1 предсказание для каждого типа регрессии (линейной и кубической).
# Найти среднее и дисперсию по предсказанной цене для каждого типа регрессии. Сделать выводы.

import numpy as np
import matplotlib.pyplot as plt

def plot_result(X,Y, lin_model=None, extra_point=None):
    plt.xlabel('size,m2')
    plt.ylabel('price,$')
    plt.plot(X, Y, 'bo') # b-blue, o-dot, ro, gx
    if lin_model:
        b = lin_model[0]
        k = lin_model[1]
        t = np.arange(X.min(), X.max(), 0.01)
        plt.plot(t, k*t+b, 'k')
        if extra_point:
            y = k*extra_point+b
            plt.plot(extra_point, y, 'ro')
    plt.grid()
    plt.show()

def normalization(X):
    mean = X.mean()
    std = X.std()
    normal = (X - mean)/std
    return normal,mean,std

def error_function(X,Y,lin_model):
    b = lin_model[0]
    k = lin_model[1]
    J = (1/len(X)) * np.sum(((k*X + b) - Y)**2)
    return J

def gradient_desceent(X,Y, alpha, n_steps, lin_model, is_printed=False):
    b = lin_model[0]
    k = lin_model[1]
    J_pred = 0
    for s in range(n_steps):
        db = (1/len(X)) * np.sum((k*X + b) - Y)
        dk = (1/len(X)) * np.sum(((k*X + b) - Y)*X)
        b -= alpha* db
        k -= alpha* dk
        J = error_function(X,Y,[b,k])
        # if is_printed or (s % 10000) == 0:
            # print(s,' error_function', J,' difference ',J-J_pred,'\nModel',b,k,' db ,dk ',db,dk,'\n')
            # J_pred = J
    return[b,k]

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
        # if is_printed or (s % 10000) == 0:
            # print(s,' error_function3', J,' difference ',J-J_pred,'\nModel',b,k,k2,k3,'\n db, dk, dk2, dk3 ',db,dk,dk2,dk3,'\n')
            # J_pred = J
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

data_set = np.append(data_set,np.zeros([len(data_set),2]),1)
print(data_set)
print(len(data_set))
predict_x = 71
for i,line in enumerate(data_set):
    print('i=',i)
    X = np.concatenate((data_set[0:i], data_set[i+1:]), axis=0)[:, 0]
    Y = np.concatenate((data_set[0:i], data_set[i+1:]), axis=0)[:, 1]
    X, x_mean, x_std = normalization(X)
    Y, y_mean, y_std = normalization(Y)

    # линейная регрессия
    start_lin_model = [0.1, 1]
    lin_model = gradient_desceent(X, Y, 0.0001, 100000, start_lin_model)
    predict_y = (lin_model[1] * (predict_x - x_mean) / x_std + lin_model[0]) * y_std + y_mean

    # регрессия полиномом 3й степени
    start_poly3model = [0.1, 1, 0.01, 0.01]
    poly3model = gradient_desceent3(X, Y, 0.0001, 100000, start_poly3model)
    x_temp = (predict_x - x_mean) / x_std
    predict_y3 = (poly3model[3] * x_temp ** 3 + poly3model[2] * x_temp ** 2 +
                 poly3model[1] * x_temp + poly3model[0]) * y_std + y_mean

    line[2] = predict_y
    line[3] = predict_y3

print(data_set)
print('линейная регрессия. среднее {} , дисперсия {}'.format(int(np.mean(data_set[:,2])),int(np.var(data_set[:,2]))))
print('регрессия полиномом 3й степени. среднее {} , дисперсия {}'.format(int(np.mean(data_set[:,3])),int(np.var(data_set[:,3]))))