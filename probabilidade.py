import numpy as np
from math import factorial
import matplotlib.pyplot as plt

x = np.array([0,33,33+42])
px = np.array([0.53,0.41,0.06])

def e(x,px):
    return sum(x*px)

def v(x,px):
    return e(x*x,px)-e(x,px)*e(x,px)

def C(a,b):
    return factorial(a)/(factorial(b)*factorial(a-b))

def binomial(x,n,p):
    return (factorial(n)/(factorial(x)*factorial(n-x)))*pow(p,x)*pow((1-p),(n-x))

def geometrica(x,p):
    return pow((1-p),(x-1))*p

def hipergeometrica(N,n,K,k):
    return C(K,k)*C((N-K),(n-k))/C(N,n)

def poisson(k,l):
    return np.exp(-l)*pow(l,k)/factorial(k)

def exponencial(x,l):
    return l*np.exp(-l*x)


exponencial(2,0.24)









