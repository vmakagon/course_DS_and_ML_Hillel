import numpy as np
from sympy import diff, symbols, sin, cos, exp, factorial

def get_polynom(f,x,n):
    k = f.subs(x, 0)
    f_polynom = k
    for i in range(1, n):
        f = diff(f)
        k = f.subs(x, 0) / factorial(i)
        f_polynom = f_polynom + k * x ** i
    return f_polynom

n = 5
x= symbols('x')
f1,f2,f3 = sin(x),cos(x),exp(x)

print(get_polynom(f1,x,n))
print(get_polynom(f2,x,n))
print(get_polynom(f3,x,n))


import matplotlib.pyplot as plt

def f_plot(f,f0, x, n):
    fig = plt.figure(figsize=(7, 7))
    plt.xlim(-7, 7)
    plt.ylim(-3, 3)
    y = np.array([get_polynom(f, x, n).subs(x, i) for i in xl])
    plt.plot(xl, y, label='Polynom ' + str(n) + ' = ' + str(get_polynom(f, x, n)))
    plt.plot(xl, f0, label=str(f))
    plt.legend()
    plt.grid()
    plt.show()

m = 600
xl = np.linspace(-m,m,100)
xl = xl/100

f_plot(f1,np.sin(xl), x, n)
f_plot(f2,np.cos(xl), x, n)
f_plot(f3,np.exp(xl), x, n)

