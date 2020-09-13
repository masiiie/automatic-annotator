import os  
import pandas as pd                

def var1(dataset, method, max_audio_by_tonema, args):
    for entonema in os.listdir(dataset):
        if os.path.isdir('{}/{}'.format(dataset,entonema)):
            for wav in os.listdir('{}/{}'.format(dataset,entonema))[:max_audio_by_tonema]:
                method(dataset, entonema, wav, *args)


def var2(dataset, method, args):
    for entonema in os.listdir(dataset):
        if os.path.isdir('{}/{}'.format(dataset,entonema)):
            for wav in os.listdir('{}/{}'.format(dataset,entonema)):
                if os.path.isdir('{}/{}/{}'.format(dataset,entonema, wav)):
                    for augmentation in os.listdir('{}/{}/{}'.format(dataset, entonema, wav)):
                        method(dataset, entonema, '{}/{}'.format(wav,augmentation), args)


def make_to_given_dataset(dataset, method, args = [], v1 = True, max_audio_by_tonema = 1000):     
    if v1:
        var1(dataset, method, max_audio_by_tonema, args)
    else:
        var2(dataset, method, args)


def write_excel(dataframe, path):
    writer = pd.ExcelWriter(path, engine='xlsxwriter')
    dataframe.to_excel(writer, sheet_name='Sheet1')
    writer.save() 

def get_data(xlsx, mode_tuple = False):
    data = pd.read_excel(xlsx)
    data.head(3)    
    #features = list(data.dtypes.index)
    #input('daframe_len = {}'.format(len(data)))

    '''
    if mode_tuple:
        y = data['entonema']
        y = pd.DataFrame(list(map(str,y.to_numpy())))
        del data['entonema']
        X = data.loc[:, data.columns[2:]]
        return X, y
    '''
    return data

def same_entonema(data, count):
    if 'entonema' in data.columns:
        gb = data.groupby('entonema')
    elif 'entonema_x' in data.columns:
        gb = data.groupby('entonema')
        
    dataframe = pd.DataFrame()
    for ent, _ in gb:
        fr = data[data.entonema == ent]
        junior = pd.DataFrame(fr.to_numpy(), columns=fr.columns).loc[:count - 1]
        dataframe = dataframe.append(junior, ignore_index=True, sort = False)

    return dataframe


def delete_items_common(data1, data2):
    df = set(data1.wav).difference(set(data2.wav))
    data1 = data1[data1.wav.isin(df)]
    return data1