import numpy as np
import numdifftools as nd
from scipy import interpolate


# Main
def spline_interpolation(y):        
    x = [t for t in np.arange(0, len(y)*np.pi/50, step=np.pi/50)]
    #x = [t for t in np.arange(0, len(y))]
    x = x[:len(y)]  # parchetazote

    if len(y) == 0:
        return (0,[],[],0)

    #print('y_len = {}\n\n'.format(len(y)))
    f = np.array([[x_,y_] for x_,y_ in zip(x,y) if y_ != 0]) 
    x_clean = f[:,0]
    y_clean = f[:,1]

    try:
        tck = interpolate.splrep(x_clean, y_clean, s=0)
        ynew = interpolate.splev(x, tck, der=0)
    except:
        print('Mismo error de "TypeError: m > k must hold"')
        ynew = []
        tck = None
    #return tck, x, ynew,  [x[0]-0.5, x[-1]+0.5, min(y_clean), max(y)]
    return ynew



'''
def second_derivate(y, interpolate = True):
    x = [t for t in np.arange(0, len(y)*0.01, step=0.01)]
    fun = y
    x = x[:len(y)]  # parchetazote

    if interpolate:
        interpolador_lineal = interpolate.interp1d(x, y, bounds_error=False, fill_value=0)
        fun = lambda p: interpolador_lineal(p[0])

    H = nd.Hessian(fun)

    return [H([t])[0][0] for t in x], x
'''

def second_derivate(y):
    tck, x, _, _ = spline_interpolation(y)
    return x, interpolate.splev(x, tck, der=2) 

def convexity(x):
    return 'convexo' if x > 0 else 'concavo' if x < 0 else 'recta'

def say_conexity(function):
    abscisas, sd = second_derivate(function)
    
    intervalos = []
    start = 0

    for i, x in enumerate(sd[1:]):
        if convexity(x) != convexity(sd[start]):
            intervalos.append((start,i, convexity(sd[start])))
            start = i

    intervalos.append((start,len(function) - 1, convexity(sd[start])))
    #input('intervalos = {}\n\n'.format(intervalos))
    return intervalos, sd, abscisas