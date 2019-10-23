import urllib.request
from bs4 import BeautifulSoup

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
        if grivna in e_price:
            price = int(price // kurs)
        data_set.append([area, price])
    except:
        pass

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
        if is_printed or (s % 10000) == 0:
            print(s,' error_function', J,' difference ',J-J_pred,'\nModel',b,k,' db ,dk ',db,dk,'\n')
            J_pred = J
    return[b,k]

data_set = np.array(data_set)
X = data_set[:,0]
Y = data_set[:,1]
plot_result(X,Y)
X, x_mean, x_std = normalization(X)
Y, y_mean, y_std = normalization(Y)

start_lin_model = [0.1,1]
plot_result(X,Y,start_lin_model)
print(error_function(X,Y,start_lin_model))
lin_model = gradient_desceent(X,Y, 0.0001, 100000, start_lin_model)
print(lin_model)
print(error_function(X,Y,lin_model))
predict_x = 71
predict_y = (lin_model[1]*(predict_x - x_mean)/x_std + lin_model[0])*y_std + y_mean
print("Прогноз цены для квартиры площадью {}м2 - {}$ (линейная регрессия)".format(predict_x, int(predict_y)))
plot_result(X,Y,lin_model,(predict_x - x_mean)/x_std)


