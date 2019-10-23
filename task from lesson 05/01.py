import keras
print(keras.__version__)

from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# from 28x28 to 784
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)

# from int to float
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

import matplotlib.pyplot as plt
import numpy as np

def show_n(nrows,ncols,img_list):
    fig, subplots = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 6))
    for i in range(nrows * ncols):
        if i == len(img_list):
            break
        ax = fig.axes[i]
        # ax.set(title=title)
        ax.imshow(img_list[i])
    plt.show()

print('Произвольный выбор эталонов цифр :')
my_digits = [407, 409, 400, 402, 405, 406, 439, 410, 401, 414]
print(my_digits)
show_n(2,5,[x_test[i].reshape(28,28) for i in my_digits])

total_correct = 0
for i, ex in enumerate(x_test):
    results = []
    for my in my_digits:
        results.append((np.dot((ex - x_test[my]),(ex - x_test[my]))**0.5))
    min_arg = np.array(results).argmin()
    if min_arg == y_test[i]:
        total_correct += 1
print(total_correct/len(y_test), ' accuracy test')

total_correct = 0
for i, ex in enumerate(x_train):
    results = []
    for my in my_digits:
        results.append((np.dot((ex - x_test[my]),(ex - x_test[my]))**0.5))
    min_arg = np.array(results).argmin()
    if min_arg == y_train[i]:
        total_correct += 1
print(total_correct/len(y_train), ' accuracy train')
print('Выбранные эталоны показывают точность меньше выбранным на занятии.')

print('\nПолучим собирательные образы (эталоны) цифр')
list_itog = []
for dig in range(10):
    list_dig = [i for i in range(len(y_test)) if y_test[i]== dig]
    s = x_test[list_dig[0]]
    for value in list_dig[1:]:
        s = s + x_test[value]
    s = s / len(list_dig)
    s_int = s.astype('uint8')
    list_itog.append(s_int)

show_n(2,5,[list_itog[i].reshape(28,28) for i in range(len(list_itog))])
my_digits = list_itog

total_correct = 0
for i, ex in enumerate(x_test):
    results = []
    for my in my_digits:
        results.append((np.dot((ex - my),(ex - my))**0.5))
    min_arg = np.array(results).argmin()
    if min_arg == y_test[i]:
        total_correct += 1
print(total_correct/len(y_test), ' accuracy test')

total_correct = 0
for i, ex in enumerate(x_train):
    results = []
    for my in my_digits:
        results.append((np.dot((ex - my),(ex - my))**0.5))
    min_arg = np.array(results).argmin()
    if min_arg == y_train[i]:
        total_correct += 1
print(total_correct/len(y_train), ' accuracy train')
print('Собирательные образы показывают хорошую точность на тестовой и на тренировочной выборке.')

