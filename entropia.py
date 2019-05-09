from math import log

x1 = [0.1, 0.2, 0.3, 0.4]
x2 = [0.25, 0.25, 0.25, 0.25]

def H(x):
    H = 0;
    for xi in x:
        H += xi*log(1/xi, 2)
    return H

def D(x, y):
    D = 0;
    for i in range(len(x)):
        D += x[i]*log(x[i]/y[i], 2)
    return D

print(D(x1, x2))
print(D(x2, x1))