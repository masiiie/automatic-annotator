import numpy as np
import os, pysptk
import matplotlib.pyplot as plt
from scipy.io import wavfile
from convexidad_spline import spline_interpolation


import amfm_decompy.pYAAPT as pYAAPT
import amfm_decompy.basic_tools as basic

def remove_silence(signal, *args):
    stp = 2
    for i in np.arange(0, len(signal), step=stp):
        if signal[i] != 0 and signal[i+stp]!=0:
            break


    for j in np.arange(len(signal) - 1, 1, step=-stp):
        if signal[j]!= 0 and signal[j-stp]!=0:
            break

    '''
    print('signal_len = {}   i={} j={}'.format(len(signal),i,j))
    a = signal[:-j]
    return a[i:]
    '''
    return signal[i:j], i, j




def get_pitch_decompy_values(wav, remove_silencess = True, interpolate = True):
    #print('get_pitch_decompy_values\npath = {}\n'.format(wav))
    signal = basic.SignalObj(wav)
    '''
    plt.title('signal')
    plt.plot(signal.data, color = 'm')
    '''
    pitch = pYAAPT.yaapt(signal)
    
    ynew = pitch.samp_values

    rv = len(pitch.samp_values)
    sv = 0
    iv = 0
    
    # toma el pitch quita los silencios del inicio y el final y aplica
    # spline_interpolation para quitar rellenar por interpolacion los silencios intermedios
    if remove_silencess:
        ynew, _, _ =  remove_silence(ynew)
        sv = len(ynew)

    #print('  y_before_remove_silence = {}'.format(len(pitch.samp_values)))
    #print('  y_to_spline_len = {}'.format(len(ynew)))
    if interpolate:
        #print('     interpolating')
        #_, _, ynew, _ = spline_interpolation(ynew)
        ynew = spline_interpolation(ynew)
        iv = len(ynew)
    
    return ynew

def get_pitch_decompy_int(wav):
    signal = basic.SignalObj(wav)
    pitch = pYAAPT.yaapt(signal)
    return remove_silence(pitch.samp_interp)

def get_pitch_pysptk(wav):
    sample_rate, samples = wavfile.read(wav)
    f0_swipe = pysptk.swipe(samples.astype(np.float64), fs=sample_rate, hopsize=80, min=60, max=200, otype="f0")
    return f0_swipe


def main():   
    for wav in os.listdir('./entonemas'):
        if os.path.isdir('./entonemas/{}'.format(wav)):
            continue
        decompy_pitch = get_pitch_decompy_values('./entonemas/{}'.format(wav))
        pysptk_pitch = get_pitch_pysptk('./entonemas/{}'.format(wav))

        fig, axarr = plt.subplots(2, figsize=(20,16))

        axarr[0].plot(decompy_pitch)
        axarr[0].set_title("decompy", loc='left')
        axarr[1].plot(pysptk_pitch)
        axarr[1].set_title("pysptk", loc='left')
        fig.suptitle('Pitches {}'.format(wav))

        '''
        plt.plot(decompy_pitch, label = 'decompy', color = 'r')
        plt.plot(pysptk_pitch, label = 'pysptk', color = 'b')
        plt.title('{} pitch'.format(wav))
        plt.legend()
        '''
        
        plt.savefig('./entonemas/compared pitches/{}_pitches.png'.format(wav))
        #plt.show()

#main()