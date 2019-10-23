# 6. Посчитайте производную функции e^(2x) + x^3 + 3
from sympy import diff, symbols, exp

x = symbols('x')

print(diff((exp(2*x)+pow(x,3)+3),x))
# print()