import torch
import matplotlib.pyplot as plt

def normalization(x):
    x_mean = x.mean()
    x_std = x.std()
    return (x - x_mean)/x_std,x_mean,x_std

def plot_result(x, y, model=None, extra_point=None):
    if model:
        bt = float(model[0])
        kt = float(model[1])
        t = torch.arange(x.min(), x.max(), 0.01)
    plt.xlabel('Price')
    plt.ylabel('No / Yes')
    plt.plot([v for i, v in enumerate(x) if y[i]==1], [v for v in y if v==1], 'bx')
    plt.plot([v for i, v in enumerate(x) if y[i]==0], [v for v in y if v==0], 'gx')
    if model:
        plt.plot(t.numpy(), 1/(1+torch.exp(-(kt*t+bt)).numpy()),color='cyan')
        if extra_point:
            y = 1/(1+torch.exp(-(kt*extra_point+bt)))
            print(' predict',float(y))
            plt.plot(extra_point, y, 'ro')
    plt.grid()
    plt.show()

x = torch.tensor([18000, 16000, 19000, 21000, 16500, 18700, 21300],dtype=torch.float32)
y = torch.tensor([0, 1, 0, 0, 1, 1, 0],dtype=torch.float32)
x,x_mean,x_std = normalization(x)
b, k = torch.tensor([0.2], requires_grad=True), torch.tensor([0.1], requires_grad=True)
plot_result(x,y,[b,k])
z = k*x + b
y_ = 1/(1+torch.exp(-z))
BCE = torch.nn.BCELoss()
loss = BCE(y_,y)
alpha = 0.01
optimazer = torch.optim.SGD([k, b], lr=alpha)
for _ in range(1001):
    z = k * x + b
    y_ = 1 / (1 + torch.exp(-z))
    loss = BCE(y_,y)
    optimazer.zero_grad()
    loss.backward()
    optimazer.step()

print('b = {}, k = {} \nloss = {}'.format(float(b),float(k),float(loss)))
x_ = (20000. - x_mean)/x_std
plot_result(x,y,[b,k],x_)