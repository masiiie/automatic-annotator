import pywt, os
import matplotlib.pyplot as plt
import numpy as np
from comparing_pitches import get_pitch_decompy_values, remove_silence
from convexidad_spline import spline_interpolation, say_conexity
from entropy import entropy, get_cD
from scipy.signal import convolve


def decorator(function):
    def sol(dataset, entonema, wav, *args):
        if wav.endswith('.wav'):
            curve_path = '{}/{}/{}'.format(dataset, entonema, wav)
            curve  = get_pitch_decompy_values(curve_path, remove_silencess = args[0], interpolate = args[1])
            if len(curve) > 0:
                return function(curve, entonema, wav, *args[2:])
        else:
            print('Error {} needs a .wav'.format(str(function)))
    return sol



def plot_before_after_transformation(dataset, entonema, wav):
    if not wav.endswith('.wav'):
        return

    path = '{}/{}/{}'.format(dataset, entonema, wav)

    ybefore = get_pitch_decompy_values(path, remove_silencess = False, interpolate = False)
    ynew  = get_pitch_decompy_values(path, remove_silencess = True, interpolate = True)

    fig, axarr = plt.subplots(2)
    axarr[0].grid()
    axarr[0].plot(ybefore)
    axarr[0].set_ylabel('No trans')
    axarr[0].set_xlabel('len = {}       entropy = {}'.format(len(ybefore), entropy(ybefore)))
    ax = plt.axis


    axarr[1].grid()
    axarr[1].plot(ynew)
    axarr[1].set_ylabel('With trans')
    axarr[1].set_xlabel('len = {}       entropy = {}'.format(len(ynew), entropy(ynew)))
    fig.subplots_adjust(hspace=0.5)
    plt.axis = ax


    dir = './temp/{}'.format(entonema)
    if not os.path.exists(dir):
        os.mkdir(dir)
    plt.savefig('{}/{}_{}.png'.format(dir, entonema, wav.replace('.wav','')))







@decorator
def plot_hessian_intervals(patron_values, entonema, wav):
    colores = {'convexo': 'lawngreen', 'concavo': 'deeppink', 'recta': 'salmon'}

    intervalos, segunda_derivada, x = say_conexity(patron_values)
    fig, axarr = plt.subplots(2)
    for a,b,clasif in intervalos:
        axarr[0].plot(x[a:b],patron_values[a:b], color = colores[clasif], label = clasif)
    
    handles, labels = axarr[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axarr[0].grid()
    axarr[0].legend(by_label.values(), by_label.keys())
    #axarr[0].set_title("      Pitch\n\n     entonema = {}    wav = {}".format(entonema, wav), loc='left')
    axarr[1].grid()
    axarr[1].plot(x, segunda_derivada, color = 'mediumorchid')
    axarr[1].set_title('Segunda derivada')
    fig.subplots_adjust(hspace=0.5)
    plt.show()
    plt.savefig('./curvas con sus formas/{}_{}.png'.format(entonema, wav))

    return intervalos
    

@decorator
def plot_descomposition(curve, entonema, wav, wavelet, directory):
    wavelet_name = wavelet if type(wavelet) == type('hola') else wavelet.name
    size = len(curve.data)
    cA, cD = pywt.dwt(curve, wavelet, mode='symmetric', axis=-1)

    fig = plt.figure()

    
    for i in np.arange(0, max_level):
        ax0 = fig.add_subplot(max_level, 2, 2*i + 1)
        ax0.grid()
        #ax0.plot(xa, cA, '-o')
        ax0.plot(cA, alpha = 0.5, label = 'cA')
        plt.yticks([min(cA), (min(cA) + max(cA))/2, max(cA)], size = 6, rotation = 20)
        plt.xticks(np.arange(0,len(cA), step=len(cA)/4), size = 6)
        ax0.set_xlabel('len = {}  entropy = {}'.format(len(cA), entropy(cA)), size = 8)
        ax0.set_ylabel("L{}".format(i), size = 8)
        #plt.setp(ax0.get_xticklabels(), fontsize=6)


        ax1 = fig.add_subplot(max_level, 2, 2*i + 2)
        ax1.grid()
        #ax1.plot(xd, cD, '-o')
        ax1.plot(cD, label = 'cD')
        #plt.yticks(np.arange(min(cD),max(cD), step=(max(cD) - min(cD))/2), size = 6)
        plt.yticks([min(cD), (min(cD) + max(cD))/2, max(cD)], size = 6, rotation = 20)
        plt.xticks(np.arange(0,len(cD), step=len(cD)/4), size = 6)
        ax1.set_xlabel('len = {}  entropy = {}'.format(len(cD), entropy(cD)), size = 8)

        if i ==0:
            ax0.set_title('cA', size = 12)
            ax1.set_title('cD', size = 12)

        cA, cD = pywt.dwt(cA, wavelet, mode='symmetric', axis=-1)
        #xa = [t for t in np.arange(0, len(cA)*np.pi/50, step=np.pi/50)]
        #xd = [t for t in np.arange(0, len(cD)*np.pi/50, step=np.pi/50)]
        



    fig.subplots_adjust(hspace=1.2, wspace=0.2)
    plt.savefig('{}/{}_{}_{}.png'.format(directory, entonema, wav.replace('.wav',''), wavelet_name))


@decorator
def plot_waverec(curve, entonema, wav, wavelet, directory):
    levels = 4
    coeff = get_cD(curve, wavelet, level = levels)

    fig = plt.figure()
    ax0 = fig.add_subplot(levels + 2, 1, 1)
    ax0.plot(curve)
    plt.yticks([min(curve), (min(curve) + max(curve))/2, max(curve)], size = 6)
    plt.xticks(np.arange(0,len(curve), step=len(curve)/10), size = 6)
    cv = convolve(curve,curve, mode='valid')[0]
    ax0.set_xlabel('longitud = {}  entropía = {}  convolución = 10**{}'.format(len(curve), entropy(curve), round(np.log10(cv),2)), size = 8)
    ax0.set_ylabel("Curva original", size = 8)
    ax0.grid()
    
    for i, (cA, cD) in enumerate(coeff):
        rec = pywt.waverec((cA,cD), 'db5')
        ax0 = fig.add_subplot(levels + 2, 1, i  + 2)
        ax0.plot(rec, alpha = 0.6)
        ax0.grid()

        plt.yticks([min(rec), (min(rec) + max(rec))/2, max(rec)], size = 6)
        plt.xticks(np.arange(0,len(rec), step=len(rec)/10), size = 6)
        cv = convolve(curve,rec, mode='valid')[0]
        ax0.set_xlabel('longitud = {}  entropía = {}  convolución = 10**{}'.format(len(rec), entropy(rec), round(np.log10(cv),2)), size = 8)
        ax0.set_ylabel("N{}".format(i), size = 8)


    fig.subplots_adjust(hspace=1.2, wspace=0.2)
    plt.savefig('{}/{}_{}_{}.png'.format(directory, entonema, wav.replace('.wav',''), wavelet))

def plot_spline_interpolation(dataset, entonema, wav):
    print('Hola!')
    if not wav.endswith('.wav'):
        return

    curve_path = '{}/{}/{}'.format(dataset, entonema, wav)
    curve  = get_pitch_decompy_values(curve_path, remove_silencess = True, interpolate = False)

    ynew = spline_interpolation(curve)
    x = np.arange(0,len(curve))
    
    plt.figure()
    plt.scatter(x, curve, s = np.pi, label='Original_pitch  entropy = {}'.format(entropy(curve)))
    plt.plot(x, ynew, label='Spline_interpolation   entropy = {}'.format(entropy(ynew)), alpha=0.4) 
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.legend(loc='upper center', fancybox=True, shadow=True, bbox_to_anchor=(0.5, 1.13))
    plt.grid()

    #plt.title('Cubic-spline interpolation\n\n    entonema = {}    wav = {}'.format(entonema, wav))
    #plt.show()
    dir = './temp/{}'.format(entonema)
    if not os.path.exists(dir):
        os.mkdir(dir)
    plt.savefig('{}/{}_{}.png'.format(dir, entonema, wav.replace('.wav','')))


@decorator
def plot_aproximation(curve, entonema, wav):
    abcisas = [x for x in np.arange(0,len(curve))]
    cA, cD = pywt.dwt(curve, 'haar', mode='symmetric', axis=-1)

    fig, axarr = plt.subplots(2)
    axarr[0].plot(curve, color = 'c')
    axarr[0].set_title('entonema = {}   wav = {}'.format(entonema, wav), loc='left')

    #axarr[1].scatter(abcisas[0: len(cA)], cA)
    axarr[1].plot(cA, color = 'r')
    axarr[1].set_title('aproximation', loc='left')
    fig.subplots_adjust(hspace=0.5)
    plt.show()