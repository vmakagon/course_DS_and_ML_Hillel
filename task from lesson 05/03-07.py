import urllib.request
from bs4 import BeautifulSoup
import numpy as np
import matplotlib.pyplot as plt

url = "https://www.lun.ua/%D0%BF%D1%80%D0%BE%D0%B4%D0%B0%D0%B6%D0%B0-%D0%BA%D0%B2%D0%B0%D1%80%D1%82%D0%B8%D1%80-%D0%BA%D0%B8%D0%B5%D0%B2?builtYearMin=2010&floorCountMin=25&isNotFirstFloor=1&isNotLastFloor=1&subway=28&subwayDistanceMax=250&withoutRenovationOnly=1"
response = urllib.request.urlopen(url)
response_text = response.read()
response.close()
print(len(response_text),response_text)
bsObj = BeautifulSoup(response_text,features="html.parser")
list = bsObj.find_all('article')
kurs = 26.5
grivna = 'грн'
data_set = []
for n,elem in enumerate(list):
    try:
        e_price = str(elem.find(class_='jss162').text)
        price = int(''.join([i for i in e_price if i in '0123456789']))
        e_price1m = str(elem.find(class_='jss163').text)
        price1m = int(''.join([i for i in e_price1m if i in '0123456789']))
        area = int(price // price1m)
        room_count = int(str(elem.find(class_='jss166').text).lstrip()[0])
        if room_count == 2:
            room_count = 0
        elif room_count == 3:
            room_count = 1
        else:
            continue
        data_set.append([area, room_count])
    except:
        pass

def normalize(X):
    return (X - X.mean())/X.std(), X.mean(), X.std()

def plot_result(X, Y, model=None, extra_point=None):
    if model:
        b = model[0]
        k = model[1]
        t = np.arange(X.min(), X.max(), 0.01)
    plt.xlabel('Area')
    plt.ylabel('3-rooms/ 2-rooms')
    plt.plot([v for i, v in enumerate(X) if Y[i]==1], [v for v in Y if v==1], 'bx')
    plt.plot([v for i, v in enumerate(X) if Y[i]==0], [v for v in Y if v==0], 'gx')
    if model:
        plt.plot(t, 1/(1+np.exp(-(k*t+b))),color='cyan')
        if extra_point:
            y = 1/(1+np.exp(-(k*extra_point+b)))
            print(' predict',y)
            plt.plot(extra_point, y, 'ro')
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

# Собрал данные с двух страниц и зафиксировал
data_set =  [[ 93, 1],
             [ 76, 0],
             [ 91, 0],
             [ 92, 1],
             [ 90, 1],
             [ 96, 1],
             [ 91, 1],
             [ 76, 0],
             [ 63, 0],
             [ 91, 1],
             [ 93, 1],
             [ 61, 0],
             [ 62, 0],
             [ 70, 0],
             [ 76, 0],
             [ 93, 1],
             [ 91, 1],
             [ 76, 0],
             [ 72, 0],
             [ 76, 0],
             [ 92, 1],
             [ 62, 0],
             [186, 0],
             [ 96, 1],
             [ 63, 0],
             [ 93, 1],
             [ 76, 0],
             [ 76, 0],
             [186, 0],
             [ 62, 0],
             [ 62, 0],
             [ 76, 0],
             [ 76, 0],
             [ 91, 1],
             [ 72, 0],
             [ 91, 1],
             [ 93, 1],
             [ 93, 1],
             [ 91, 1],
             [ 91, 1],
             [ 92, 1],
             [ 96, 1],
             [ 96, 1],
             [ 92, 1],
             [ 91, 1],
             [ 91, 1],
             [ 91, 1],
             [ 93, 1],
             [ 91, 1],
             [ 91, 1],
             [ 92, 1],
             [ 61, 0],
             [ 76, 0]]
data_set = np.array(data_set)
X = data_set[:,0]
Y = data_set[:,1]
print(len(X),X)
print(len(Y),Y)
plt.hist(X)
plt.show()
plt.boxplot(X,vert=False)
plt.show()
print('Есть аномальные выбросы. Нужно удалить.')
X0_std = data_set[:,0].std()
X0_mean = data_set[:,0].mean()
list_index_outliers = []
for i,value in enumerate(data_set):
    if not ((X0_mean - 1.96*X0_std) < value[0] < (X0_mean + 1.96*X0_std)):
        list_index_outliers.append(i)
for i in list_index_outliers[::-1]:
    data_set = np.delete(data_set,i,axis=0)
    X = np.delete(X,i,axis=0)
    Y = np.delete(Y,i,axis=0)
plt.hist(X)
plt.show()
plt.boxplot(X, vert=False)
plt.show()

X_, X_mean, X_std = normalize(X)
lin_model_start = [0, 5]
plot_result(X_, Y, lin_model_start)
X = np.array([[1] + [v.tolist()] for v in X_])

print(binary_crossentropy(X, Y, [0, 1]), binary_crossentropy(X, Y, [0, -1]))
print(binary_crossentropy(X, Y, [0, 3]), binary_crossentropy(X, Y, [0, -3]))
print(binary_crossentropy(X, Y, [0, 5]), binary_crossentropy(X, Y, [0, -5]))
print(binary_crossentropy(X, Y, [0, 6]), binary_crossentropy(X, Y, [0, -10]))
print(binary_crossentropy(X, Y, [1, 1]), binary_crossentropy(X, Y, [-1, -1]))
print(binary_crossentropy(X, Y, [1, 3]), binary_crossentropy(X, Y, [-1, -3]))
print(binary_crossentropy(X, Y, [1, 5]), binary_crossentropy(X, Y, [-1, -5]))
print(binary_crossentropy(X, Y, [1, 10]), binary_crossentropy(X, Y, [-1, -10]))



print('binary_crossentropy start',binary_crossentropy(X, Y, lin_model_start))
lin_model = gradient_desceent(X,Y, 0.0001, 100_000, lin_model_start)
print('binary_crossentropy the best',binary_crossentropy(X, Y, lin_model))
print('lin_model ',lin_model)
predict = 72
print('for',predict,'m2 ',end='')
plot_result(X_, Y, lin_model, (predict - X_mean)/X_std)

np.shape