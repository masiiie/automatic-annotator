import numpy as np
import pandas, pywt, os
from entropy import entropy, get_cD, cD_entropy_vectorizer, frecuency_features
from comparing_pitches import get_pitch_decompy_values
from scipy.signal import convolve
from utils import write_excel

class Extra_info_vectorizer:
    def __init__(self, remove_silences =  True, interpolate = True):
        self.max = 0
        self.silences = remove_silences
        self.interpolate = interpolate

    def get_features_names(self):
        spectrum_ = ['sptr{}'.format(i) for i in range(self.max)]
        features_names = ['pendiente_pitch'] + spectrum_
        return features_names

    def info(self, dataset, entonema, wav, curve = []):
        wav_path = '{}/{}/{}'.format(dataset,entonema,wav)
        if len(curve) == 0:
            curve = get_pitch_decompy_values(wav_path, \
            remove_silencess = self.silences, interpolate = self.interpolate)
        pendt = pendiente_pitch(curve)
        spectrumm = spectrum(curve)
        if(len(spectrumm) > self.max): self.max = len(spectrumm)
        return [pendt] + spectrumm.tolist()



class Info_vectorizer:
    def __init__(self, remove_silences =  True, interpolate = True):
        self.silences = remove_silences
        self.interpolate = interpolate
    def get_features_names(self):
        cA_entr_ = ['entr_cA{}'.format(i) for i in range(0,5)]
        cD_entr_ = ['entr_cD{}'.format(i) for i in range(0,5)]
        conv_ = ['conv{}'.format(i) for i in range(0,5)]
        features_names = ['entr_pitch'] + cA_entr_ + cD_entr_ + conv_
        return features_names
            

    def info(self, dataset, entonema, wav, curve = []):
        wav_path = '{}/{}/{}'.format(dataset,entonema,wav)
        if len(curve) == 0:
            curve = get_pitch_decompy_values(wav_path, \
            remove_silencess = self.silences, interpolate = self.interpolate)
        entr_pitch = [round(entropy(curve), ndigits=2)]
        cA_entr = []
        cD_entr = []
        conv = []

        descp = get_cD(curve, 'db5', level = 4)
        for (cA, cD) in descp:
            cA_entr.append(round(entropy(cA), ndigits=2))
            cD_entr.append(round(entropy(cD), ndigits=2))
            rec = pywt.waverec((cA,cD), 'db5')
            cv = convolve(curve,rec, mode='valid')[0]
            conv.append(round(cv,ndigits=2))

        return entr_pitch + cA_entr + cD_entr + conv


def make_statistic(dataset, info_vectorizer, max_wav_by_tonema = 1000):
    data = []
    for entonema in os.listdir(dataset):
        print('entonema = {}'.format(entonema))
        if os.path.isdir('{}/{}'.format(dataset, entonema)):
            for wav in os.listdir('{}/{}'.format(dataset,entonema))[:max_wav_by_tonema]:
                if wav.endswith('.wav'):
                    try:
                        basics = [wav, entonema]
                        vector = info_vectorizer.info(dataset, entonema, wav)
                        data.append(basics + vector)
                    except:
                        print('Failed  {}  {}'.format(entonema, wav))


    basics = ['wav', 'entonema']
    extras = info_vectorizer.get_features_names()

    dataframe = pandas.DataFrame(data, columns = basics + extras).fillna(value = 0).round(2)

    write_excel(dataframe, './features.xlsx')



# Extras


def pendiente_pitch(pitch):
    x = np.arange(len(pitch))
    regr = np.polyfit(x, pitch, 1)
    return regr[0]




def spectrum(pitch):
    sft = np.fft.fft(pitch)
    spectrum = np.abs(sft)
    return spectrum

def naive_vectorizer(curve, true_features = 0):
        '''
                columns:
        Index(['cD0', 'cD1', 'cD2', 'cD3', 'cD4', 'cD5', 'cD6', 'cD7', 'cD8', 'cD9',
            'cD10', 'cD11', 'cD12', 'cD13', 'entropy_level', 'wav', 'cD14', 'cD15',
            'cD16', 'cD17', 'cD18', 'cD19', 'cD20', 'cD21', 'cD22', 'cD23', 'cD24',
            'cD25', 'cD26', 'cD27', 'cD28', 'cD29', 'cD30', 'cD31', 'entr_pitch',
            'entr_cA4', 'conv0', 'f_inicial', 'f_final', 'f_max', 'f_min',
            'pendiente_pitch', 'entonema'],
            dtype='object')
        '''
        f1 = ['cD{}'.format(i) for i in range(0, 14)]
        f2 = ['entropy_level']
        f3 = ['cD{}'.format(i) for i in range(14, 32)]
        f4 = ['entr_pitch','entr_cA4', 'conv0']
        f5 = ['f_inicial', 'f_final', 'f_max', 'f_min', 'pendiente_pitch']
        truely_features = f1 + f2 + f3 + f4 + f5
        if true_features != 0:
            truely_features = true_features
        
        vz1 = cD_entropy_vectorizer('db5', 3, 1)
        vz2 = Extra_info_vectorizer(remove_silences = True, interpolate = False)
        vz3 = Info_vectorizer(remove_silences = True, interpolate = False)

        v1, fn1 = vz1(curve)
        v2 = vz2.info(1,1,1, curve = curve)
        v3 = vz3.info(1,1,1, curve = curve)
        fn2 = vz2.get_features_names()
        fn3 = vz3.get_features_names()
        v4, fn4 = frecuency_features(curve)
        features = fn1 + fn2[:1] + fn3 + fn4
        vector = list(v1) + list(v2[:1]) + list(v3) + list(v4)

        df = pandas.DataFrame([vector], columns = features)
        df_aux = pandas.DataFrame(columns=truely_features)
        df_aux = df_aux.append(df, ignore_index = True, sort = False)
        df_aux = df_aux.loc[:, truely_features].fillna(value=0)

        #print('c:\n{}'.format(df_aux.columns))

        vector = df_aux.to_numpy()[0]   

        
        #print('vector_part1:\n{}'.format(df_aux.iloc[:,:10]))
        #print('vector_part2:\n{}'.format(df_aux.iloc[:,10:20]))
        #print('vector_part3:\n{}'.format(df_aux.iloc[:,20:30]))
        #print('vector_part4:\n{}'.format(df_aux.iloc[:,30:43]))

        
        return vector, truely_features