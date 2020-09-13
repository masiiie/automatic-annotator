import pywt
import numpy as np


# Mains
def frecuency_features(curve):
    f_inicial = curve[0]
    f_final = curve[-1]
    f_max = max(curve)
    f_min = min(curve)

    return [f_inicial, f_final, f_max, f_min], ['f_inicial', 'f_final', 'f_max', 'f_min'] 

def make_entropy_vector(curve, level = 3, wavelet_list = pywt.wavelist(kind='discrete')):
    vector = []
    features_name = []
    for wavelet in wavelet_list:
        #cA, cD = pywt.dwt(curve, wavelet, mode='symmetric', axis=-1)
        cD = get_cD(curve, wavelet, level = level)[-1][1]
        vector.append(entropy(cD))
        features_name.append('entr_{}'.format(wavelet))
    return vector, features_name



def entropy(x):
    probs = [np.mean(x == valor) for valor in set(x)]
    return round(np.sum(-p * np.log2(p) for p in probs), 3)



# Vectorizers

def special_vectorice(level):
    def sol(curve):
        #w = pywt.Wavelet('bspline3')  esto da error
        w = pywt.Wavelet('bspline3', filter_bank=wavelet_BattleLamarie(m=3))
        w.orthogonal = True
        w.biorthogonal = False
        cD = get_cD(curve, w, level = level)[-1][1]
        e = entropy(cD)
        vector = np.append(cD, [e]) 
        return vector
    return sol


# Devuelve los coeficientes (cA o cD segun v) del nivel "level" de la DWT de la wavelet dada
def cD_entropy_vectorizer(wavelet, level, cA_return):
    def sol(curve):
        cD = get_cD(curve, wavelet, level = level)[-1][cA_return]
        e = entropy(cD)
        #print('cD = {}    e = {}'.format(cD, e))
        vector = np.append(cD, [e]) 
        letra = 'A' if cA_return == 0 else 'D'
        return vector, ['c{}{}'.format(letra, i) for i in np.arange(len(cD))] + ['entropy_level']
    return sol
    

def all_levels_vectorizer(wavelet, first_level, final_level):
    def sol(curve):
        levels = get_cD(curve, wavelet, level = final_level)
        return levels[first_level:]
    return sol



## Utils

import matplotlib.pyplot as plt

def get_cD(curve, wavelet, level = 5):
    if len(curve) == 0:
        return []
    if type(wavelet) == type('HOla'):
        wavelet = pywt.Wavelet(wavelet)


    levels = []
    cA, cD = pywt.dwt(curve, wavelet, mode='symmetric', axis=-1)

    for i in np.arange(0, level + 1):
        levels.append((cA, cD))
        #levels.append(cD)
        cA, cD = pywt.dwt(cA, wavelet, mode='symmetric', axis=-1)
    return levels