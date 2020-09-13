import utils, os
import pandas as pd

def merge(features1, features2, max = 0, features3 = 0):
    data = features1.merge(features2, on = 'wav')
    if type(features3) == type(pd.DataFrame()):
        data = data.merge(features3, on = 'wav')

    if not 'entonema' in data.columns:
        data['entonema'] = data['entonema_x']

    if max != 0:
        data = utils.same_entonema(data, max)

    data.drop(data.filter(regex='_y$').columns.tolist(),axis=1, inplace=True)
    data.drop(data.filter(regex='_x$').columns.tolist(),axis=1, inplace=True)
    
    return data

def get(dataset, transform, cd_wavelet = None,\
    statistics_features = ['entr_pitch','entr_cA3','conv0'], spectrum_features = []):
    super_dict = {
            'test':
                {
                    'tf':
                    {
                        'db53': 11,
                        'statistics': 9,
                        'spectrum': 10,
                        'entropys': 12,
                        'frecuency_features': 13,
                        'haar3_ca': 14
                    },
                    True:
                    {
                        'haar3': None,
                        'db53': 5,
                        'statistics': 7,
                        'spectrum': 6
                    },
                    False:
                    {
                        'haar3': 4,
                        'db53': 1,
                        'db54': 8,
                        'statistics': 3,
                        'spectrum': 2
                    }
                },
            'definite':
                {
                    'tf':
                    {
                        'db53': 12,
                        'statistics': 14,
                        'spectrum': 13,
                        'entropys': 12,
                        'frecuency_features': 16,
                        'haar3_ca': 17
                    },
                    True:
                    {
                        'haar3': None,
                        'db53': 8,
                        'statistics': 9,
                        'spectrum': 10
                    },
                    False:
                    {
                        'haar3': 6,
                        'db53': 7,
                        'db54': 11,
                        'statistics': 4,
                        'spectrum': 5
                    }
                },

            'definite augmentation 1 samples from 2 samples':
                {
                    'tf':
                    {
                        'db53': 10,
                        'statistics': 12,
                        'spectrum': 11
                    },
                    True:
                    {
                        'haar3': None,
                        'db53': 5,
                        'statistics': 7,
                        'spectrum': 8
                    },
                    False:
                    {
                        'haar3': 2,
                        'haar4': 1,
                        'db53': 6,
                        'db54': 9,
                        'statistics': 3,
                        'spectrum': 4
                    }
                },
            'definite augmentation 5 samples':
            {
                'tf':
                    {
                        'db53': 3,
                        'statistics': 1,
                        'spectrum': 2,
                        'entropys': 4,
                        'frecuency_features': 5
                    }
            }
        }
    
    dsb = os.path.basename(dataset)

    cD = None
    if cd_wavelet == 'haar3':
        cD = utils.get_data('{}/features{}.xlsx'.format(dataset, super_dict[dsb][transform]['haar3']))
    elif cd_wavelet == 'db53':
        cD = utils.get_data('{}/features{}.xlsx'.format(dataset, super_dict[dsb][transform]['db53']))
    elif cd_wavelet == 'db54':
        cD = utils.get_data('{}/features{}.xlsx'.format(dataset, super_dict[dsb][transform]['db54']))
    elif cd_wavelet == 'haar3_ca':
        cD = utils.get_data('{}/features{}.xlsx'.format(dataset, super_dict[dsb][transform]['haar3_ca']))
        
        
    ff = utils.get_data('{}/features{}.xlsx'.format(dataset, super_dict[dsb][transform]['frecuency_features']))  
    entropys = utils.get_data('{}/features{}.xlsx'.format(dataset, super_dict[dsb][transform]['entropys']))  
    spectrum = utils.get_data('{}/features{}.xlsx'.format(dataset, super_dict[dsb][transform]['spectrum']))
    statistic = utils.get_data('{}/features{}.xlsx'.format(dataset, super_dict[dsb][transform]['statistics']))
    statistic = statistic.loc[:, statistics_features + ['wav', 'entonema']]
    if len(spectrum_features) > 0:
        spectrum = spectrum.loc[:, spectrum_features + ['wav', 'entonema']]

    '''
    print('dataset: {}'.format(dataset))
    print('cD:\n{}'.format(cD))
    print('sttistics:\n{}'.format(statistic))
    print('spectrum:\n{}'.format(spectrum))
    input(':)')
    '''

    return cD, spectrum, statistic, entropys, ff