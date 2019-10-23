from keras.datasets import mnist
import numpy as np
from collections import Counter

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# from 28x28 to 784
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)

# from int to float
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')


def dist(img1, img2):
    euclidean_distance = (np.dot((img1 - img2),(img1 - img2))**0.5)
    return euclidean_distance

def knn(img, x_train, y_train, k1=10,k2=50):
    neighbors = []
    distances = []
    for i,el in enumerate(x_train):
        distances.append(dist(img,el))
        neighbors.append(y_train[i])
    distances, neighbors = zip(*sorted(zip(distances, neighbors)))
    k1_neighbors = neighbors[:k1]
    k2_neighbors = neighbors[:k2]
    count1 = Counter(k1_neighbors)
    count2 = Counter(k2_neighbors)
    result1 = count1.most_common(1)[0][0]
    result2 = count2.most_common(1)[0][0]
    return result1,result2

start = 0
stop = 1000
x_test = x_test[start:stop]
y_test = y_test[start:stop]
total_correct10 = 0
total_correct50 = 0
for i, ex in enumerate(x_test):
    bestk10,bestk50 = knn(ex, x_train, y_train, 10, 50)
    if i % 20 == 0:
        print(i, bestk10, bestk50, y_test[i])
    if bestk10 == y_test[i]:
        total_correct10 += 1
    if bestk50 == y_test[i]:
        total_correct50 += 1

print(total_correct10/(stop - start), ' accuracy test KNN 10 on count {} (start {} : stop {})'.format(stop-start,start,stop))
print(total_correct50/(stop - start), ' accuracy test KNN 50 on count {} (start {} : stop {})'.format(stop-start,start,stop))
''' Результаты, выполнение занимает между 10-15 минутами
0.956  accuracy test KNN 10 on count 1000 (start 0 : stop 1000)
0.937  accuracy test KNN 50 on count 1000 (start 0 : stop 1000)
'''