import matplotlib.pyplot as plt
import numpy as np

X_ = np.arange(0, 32, 2)
Y = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1])

def normalize(X):
    return (X - X.mean())/X.std(), X.mean(), X.std()

def plot_result(X, Y, model=None):
    if model:
        b = model[0]
        k = model[1]
        t = np.arange(X.min(), X.max(), 0.01)
    plt.xlabel('Age')
    plt.ylabel('No/Yes')
    plt.plot([v for i, v in enumerate(X) if Y[i]==1], [v for v in Y if v==1], 'bo')
    plt.plot([v for i, v in enumerate(X) if Y[i]==0], [v for v in Y if v==0], 'ro')
    if model:
        plt.plot(t, 1/(1+np.exp(-k*t+b)))
    plt.show()

def activation(z):
    return 1/(1 + np.exp(-z))

def log_reg(X, model):
    z = np.dot(model, X)
    output = activation(z)
    return output

def binary_crossentropy(X, Y, model):
    # Функция ошибки
    J = 0
    m = len(X)
    for i, y in enumerate(Y):
        J += -(1/m)*(y*np.log(log_reg(X[i], model)) + (1-y)*np.log(1 - (log_reg(X[i], model))))
    return J

def gradient_desceent(X,Y, alpha, n_steps, lin_model, is_printed=False):
    b = lin_model[0]
    k = lin_model[1]
    J_pred = 0
    for s in range(n_steps):
        db = (1/len(X)) * np.sum((log_reg(X.transpose(), [b,k]) - Y) * X[:,0])
        dk = (1/len(X)) * np.sum((log_reg(X.transpose(), [b,k]) - Y) * X[:,1])
        b -= alpha* db
        k -= alpha* dk
        J = binary_crossentropy(X,Y,[b,k])
        if is_printed:
            if s % 1000 == 0:
                print(s,' binary_crossentropy', J,' difference ',J-J_pred,'\nModel',b,k,' db ,dk ',db,dk,'\n')
                J_pred = J
    return[b,k]

X_, X_mean, X_std = normalize(X_)
lin_model_start = [0, 4.]
plot_result(X_, Y, lin_model_start)
X = np.array([[1] + [v.tolist()] for v in X_])

print('binary_crossentropy start',binary_crossentropy(X, Y, lin_model_start))
lin_model = gradient_desceent(X,Y, 0.001, 100_000, lin_model_start,False)
print('binary_crossentropy the best',binary_crossentropy(X, Y, lin_model))
print(lin_model)
plot_result(X_, Y, lin_model)