import os, time
import pandas as pd
import numpy as np  
from comparing_pitches import get_pitch_decompy_values
from utils import get_data, write_excel



## Main
def merge_excel(final_dataset_name, excels_path):
    dataframe = pd.DataFrame()
    for excel in os.listdir(excels_path):
        if excel.endswith('.xlsx'):
            readed_data = get_data('{}/{}'.format(excels_path,excel))
            dataframe = dataframe.append(readed_data, ignore_index=True, sort=False)
    dataframe = dataframe.fillna(value=0)
    write_excel(dataframe, '{}.xlsx'.format(final_dataset_name))



def make_excels(folder, entonema, vectoricer, excels_path, max_audio_by_folder = 1000, processer = True, remove_silencess = True, interpolate = True):
    #input('processer = {}    silencess = {}'.format(processer, remove_silencess))
    processer = simply_add_vector_processer if processer else by_level_cD_processer
    #folder = os.path.join(dataset, entonema, wav)
    if os.path.isdir(folder):
        data, archivos, start = [], 0, time.time()
        dataframe = pd.DataFrame([])

        for augmentation in os.listdir(folder)[:max_audio_by_folder]: 
            if augmentation.endswith('.wav'):     
                curve = get_pitch_decompy_values('{}/{}'.format(folder, augmentation), remove_silencess = remove_silencess, interpolate = interpolate)
                dataframe = processer(dataframe, vectoricer, curve, augmentation)
                archivos += 1 
                #print('    augmentation = {}  vector_len = {}'.format(augmentation, len(vector)))
                #print(':)')
                '''
                except:
                    print('deprecated = {}     folder = {}'.format(augmentation, os.path.basename(folder)))
                    #print('    augmentation deprecated   dataset = {} entonema {} wav = {} augmentattion = {}'.format(dataset, entonema, wav, augmentation))
                '''
        dataframe['entonema'] = str(entonema)
        dataframe = dataframe.fillna(value=0)
        write_excel(dataframe, '{}/{}_{}.xlsx'.format(excels_path, entonema, ''))  



## Processers

def simply_add_vector_processer(dataframe, vectoricer, curve, wav):
    vector, features_name = vectoricer(curve)
    vector = list(vector) + [wav]
    features_name = features_name + ['wav']
    if len(vector) == 0:
        print('vector_len equal 0.')
        return dataframe
    data_vector = pd.DataFrame([vector], columns=features_name)
    dataframe = dataframe.append(data_vector, ignore_index=True, sort=False)
    #print('\n\n\ndatavector:\n{}'.format(data_vector))
    #input('\ndataframe inside:\n{}'.format(dataframe))
    return dataframe
    
def by_level_cD_processer(dataframe, vectoricer, curve, wav):
    vector = vectoricer(curve)
    if len(vector) == 0:
        print('vector_len equal 0.')
        return dataframe
    cols=['cD{}{}'.format(i, j) for i in np.arange(len(vector)) for j in np.arange(len(vector[i][1]))]
    a = []
    for x in vector:
        a = a + x[1].tolist()
    a = a + [wav]
    data_vector = pd.DataFrame([a], columns=cols + ['wav'])
    dataframe = dataframe.append(data_vector, ignore_index=True, sort=False)
    return dataframe
    



## Utils

