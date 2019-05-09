import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate

def v(x):     
    y = np.zeros(len(x))     
    for i in range(len(x)):         
        if x[i] >= 0:             
            y[i] = 1.     
    return y

def f(n):
    return pow(0.5,abs(-n-2))

def fp(n):
    return (f(n)+f(-n))/2

def fi(n):
    return (f(n)-f(-n))/2
    
def plot(i, func, title):
    plt.figure(i,figsize=(10,10))
    plt.subplot(1,1,1)
    plt.title(title)
    plt.stem(n, func(n))

  
n = np.arange(-10,10,0.01)
        
plot(1, f, 'f')
plot(2, fp, 'fp')
plot(3, fi, 'fi')
