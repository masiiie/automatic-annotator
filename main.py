import make_panda_set, statistics, svm_model, utils, data_manager
import warnings, pywt, os, time
import pandas as pd
from plotters_curve_curvename import *


from sklearn.feature_selection import VarianceThreshold, SelectKBest
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif
from sklearn.model_selection import ShuffleSplit, RepeatedStratifiedKFold

from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings('ignore') 

# recordar q cuando vectorice con la wavelet extranna el profe quiere q incluya
# la entropia y los rasgos de raquel

def wavelet_descomposition_experiment():
    dataset = '../datasets/grabado como el libro super bien'
    experiment_rsult = './descomposition experiment con entonema 6'
    if not os.path.exists(experiment_rsult):
        os.mkdir(experiment_rsult)

    name = os.path.basename(dataset)
    for family in pywt.families():
        wavelet = pywt.wavelist(family=family, kind='discrete')[0]
        if wavelet in pywt.wavelist(kind='discrete'):    
            directory = '{}/{}'.format(experiment_rsult, wavelet)
            if not os.path.exists(directory):
                os.mkdir(directory)
            for wav in os.listdir('{}/6'.format(dataset)):
                plot_descomposition(dataset, 6, wav, wavelet, directory)
    print('Finish')






def battle_lemarie_experiment():
    wavelet = pywt.Wavelet('bspline3', filter_bank=wavelet_BattleLamarie(m=3))
    #input('o = {}   b = {}'.format(wavelet.orthogonal, wavelet.biorthogonal))
    wavelet.orthogonal = True
    wavelet.biorthogonal = False
    directory = './descomposition {} {}'.format(wavelet.name, os.path.basename(dataset))
    os.mkdir(directory)

    utils.make_to_given_dataset(dataset, plot_descomposition, [wavelet, directory, dataset], v1 = False)





def entropy_vectorizer_svm_model_experiment():
    xlsx = '../datasets excel/entropy vector preporcessed/data2.xlsx'
    svm_model.svm_work(xlsx)

def make_xlsx_entropy_vectorizer():
    dataset = '../datasets/super dataset augmentation'
    bn_dataset = os.path.basename(dataset)
    excels_path = 'excels {}'.format(bn_dataset)
    if not os.path.exists(excels_path):
        os.mkdir(excels_path)

    utils.make_to_given_dataset(dataset, make_panda_set.make_excels, args = [entropy.make_entropy_vector, excels_path])
    make_panda_set.merge_excel('dataset = {} vectorizer = {}'.format(bn_dataset, entropy.make_entropy_vector.__name__), excels_path)
    print('Finish   make_xlsx_entropy_vectorizer!')

def make_xlsx_cD_haar_entropy_vectorizer():
    dataset = '../datasets/super dataset augmentation'
    bn_dataset = os.path.basename(dataset)
    excels_path = 'excels'
    if not os.path.exists(excels_path):
        os.mkdir(excels_path)
    
    utils.make_to_given_dataset(dataset, make_panda_set.make_excels, args = [entropy.cD_entropy_vectorizer('haar', 1, True) , excels_path])
    make_panda_set.merge_excel('dataset = {} vectorizer = {}'.format(bn_dataset, entropy.special_vectorice.__name__), excels_path)
    print('Finish   {}!'.format(make_xlsx_entropy_vectorizer.__name__))


def make_xlsx_all_levels_haar_organized():
    dataset = '../datasets/super dataset augmentation'
    start = time.time()
    bn_dataset = os.path.basename(dataset)
    excels_path = 'excels'
    if not os.path.exists(excels_path):
        os.mkdir(excels_path)            
    
    #make_panda_set.make_excels(vectoricer, excels_path, processer = True, remove_silencess, interpolate)
    vectorizer = entropy.all_levels_vectorizer('haar',3,5)
    args = [vectorizer, excels_path, False, False, False]
    utils.make_to_given_dataset(dataset, make_panda_set.make_excels, args = args)
    make_panda_set.merge_excel('vectorizer = {}'.format(vectorizer.__name__), excels_path)
    print('time making xlsx = {}mins'.format((time.time() - start)/60))
    print('Finish!')


#make_xlsx_all_levels_haar_organized()
#svm_model.svm_work('../datasets excel/8-/data8.xlsx')

'''
start = time.time()
utils.make_to_given_dataset('../audios/definite', plot_before_after_transformation, max_audio_by_tonema = 5)
print('delay = {}mins'.format((time.time() - start)/60))
'''



'''
start = time.time()
path = "./experimentos/8-"

for sample in os.listdir(path):
    if os.path.isdir('{}/{}'.format(path,sample)):
        for file in os.listdir('{}/{}'.format(path, sample)):
            plot_before_after_transformation(path, sample, file)

print('delay = {}mins'.format((time.time() - start)/60))
'''

'''
start = time.time()
args = args = [True, True, 'db5', './temp']
utils.make_to_given_dataset('../audios/definite', plot_descomposition, args = args, max_audio_by_tonema = 5)
print('delay = {}mins'.format((time.time() - start)/60))
'''


'''
start = time.time()
utils.make_to_given_dataset('../audios/definite', plot_spline_interpolation, max_audio_by_tonema = 5)
print('delay = {}mins'.format((time.time() - start)/60))
'''

'''
start = time.time()
iv = statistics.Info_vectorizer(remove_silences = True, interpolate = False)
#ef = statistics.Extra_info_vectorizer(remove_silences = True, interpolate = False)
dataset = '../audios/definite'
statistics.make_statistic(dataset, iv)
print('delay = {}mins'.format((time.time() - start)/60))
'''
'''
start = time.time()
args = [True, False, 'db5', './temp']
utils.make_to_given_dataset('../audios/definite', plot_waverec, args = args, max_audio_by_tonema = 1)
print('delay = {}mins'.format((time.time() - start)/60))
'''


'''
from entropy import cD_entropy_vectorizer, make_entropy_vector, frecuency_features
from statistics import naive_vectorizer

start = time.time()
dataset = '../II/audios/definite'
#vectoricer = lambda curve: frecuency_features(curve)
#vectoricer = lambda curve: make_entropy_vector(curve, level = 4)
#vectoricer = cD_entropy_vectorizer('db5', 3, 1)
#vectoricer = cD_entropy_vectorizer('haar', 3, 0)
vectoricer = lambda curve: naive_vectorizer(curve)


for entonema  in os.listdir(dataset):
    dir = '{}/{}'.format(dataset, entonema)
    if os.path.isdir(dir):
        make_panda_set.make_excels(dir, entonema, \
        vectoricer, './temp',
        remove_silencess = True, interpolate = False)


make_panda_set.merge_excel('features', './temp')
print('delay = {}mins'.format((time.time() - start)/60))
'''

'''
data = utils.get_data('../audios/definite augmentation 2 samples/statistics1.xlsx')
columns = list(data.columns)[2:]
indexes = data.loc[:,'wav']
data_ = data.loc[:,columns].fillna(value=0).to_numpy()
dataframe = pd.DataFrame(data_, index =  indexes, columns= columns)
utils.write_excel(dataframe, './last.xlsx')
'''


'''
xlsx = '../models vectorizers/7- 0.7/data7.xlsx'
data = utils.get_data(xlsx)
gb = data.groupby('entonema')
dataframe = pd.DataFrame()
for ent, _ in gb:
    fr = data[data.entonema == ent]
    junior = pd.DataFrame(fr.to_numpy(), columns=fr.columns).loc[:60]
    dataframe = dataframe.append(junior, ignore_index=True, sort = False)

utils.write_excel(dataframe,'./a.xlsx')
'''


clf_list = {
    'lgr': LogisticRegression(max_iter=10000),
    'svc': SVC(decision_function_shape='ovo'),
    'lsvc3': LinearSVC(random_state=0, tol=1e-05, penalty='l2', loss='hinge', max_iter=10000),
    'lsvc2': LinearSVC(random_state=0, tol=1e-05, penalty='l2')
}

fs = lambda X,y:X


sf = ['entr_pitch','entr_cA4','conv0']
spect_f = ['pendiente_pitch']

cd1, spectrum1, statistics1, entropys1, ff1 = \
    data_manager.get('./audios/definite', 'tf', cd_wavelet = 'db53', \
    statistics_features = sf, spectrum_features = spect_f)
cd2, spectrum2, statistics2, entropys2, ff2 = \
    data_manager.get('./audios/definite augmentation 5 samples', 'tf',\
    cd_wavelet = 'db53', statistics_features = sf, spectrum_features = spect_f)
cd3, spectrum3, statistics3, entropys3, ff3 = \
    data_manager.get('./audios/test', 'tf', cd_wavelet = 'db53', \
    statistics_features = sf, spectrum_features = spect_f)


mrg1 = data_manager.merge(cd1, statistics1, features3 = ff1)
mrg2 = data_manager.merge(cd2, statistics2, features3 = ff2)

mrg1 = data_manager.merge(mrg1, spectrum1)
mrg2 = data_manager.merge(mrg2, spectrum2)

dataframe = mrg1.append(mrg2, ignore_index=True, sort=False).fillna(value = 0)

test = data_manager.merge(cd3, statistics3, features3 = ff3)
test = data_manager.merge(test, spectrum3)


## Look!!!   Tome todos los audios de definite excepto los que estan en test
df = set(dataframe.wav).difference(set(test.wav))
dataframe = dataframe[dataframe.wav.isin(df)]


aux1 = test.append(dataframe, ignore_index = True, sort = False)
aux2 = pd.DataFrame(data = None, columns = aux1.columns) 
test = test.append(aux2, ignore_index = True, sort = False).fillna(value = 0)
dataframe = dataframe.append(aux2, ignore_index = True, sort = False).fillna(value = 0)

#utils.write_excel(dataframe, './real.xlsx')

cv = ShuffleSplit(n_splits=30, test_size=0.4, random_state=0)
clf = SVC(decision_function_shape='ovo')
#clf = LogisticRegression(max_iter=10000)


#svm_model.svm_work(dataframe, clf, lambda X,y:X,cv)
svm_model.definite_score(dataframe, test, clf)



'''
entr1 = entropys1.append(entropys2, ignore_index=True, sort=False).fillna(value = 0)
entr_test = entropys3

cv = ShuffleSplit(n_splits=30, test_size=0.4, random_state=0)
clf = SVC(decision_function_shape='ovo')
svm_model.svm_work(entr1, clf, lambda X,y:X, cv)

fs_list = dict([('kbest_{}_{}'.format(fun.__name__, k),\
     lambda X, y : SelectKBest(fun,k = k).fit_transform(X, y)) \
     for fun in [f_classif, mutual_info_classif] for k in [80, 100, 200]])
fs_list = {
    'identity': lambda X, y: X,\
    'kbest_mutual_info_classif_100': \
    lambda X, y: SelectKBest(score_func=mutual_info_classif, k = 100).fit_transform(X, y)}
clf_list = {
    'lgr_reg': LogisticRegression(penalty='l1', solver='liblinear', tol=1e-6,\
    max_iter=int(1e6), warm_start=True, intercept_scaling=10000)
}
ss = pd.DataFrame()
for fs in fs_list:
    print('Feature selection: {}'.format(fs))
    scores = []
    for clf in clf_list:
        print('Model: {}'.format(clf_list[clf].__class__))
        scores.append(svm_model.definite_score(dataframe, test, clf_list[clf], fs_list[fs])) 
    dj = pd.DataFrame([scores + [fs]], columns = list(clf_list.keys()) + ['feature_selection'])
    ss = ss.append(dj, ignore_index=True, sort=False)

print('Results:\n{}'.format(ss)) 
utils.write_excel(ss, './results.xlsx')
'''
    

'''
from demo import transform

dataset = '../audios/definite'
for entonema in os.listdir(dataset):
    for wav in os.listdir('{}/{}'.format(dataset, entonema))[5:7]:
        if wav.endswith('.wav'):
            #print('ent: {}  wav: {}'.format(entonema, wav))
            transform('{}/{}/{}'.format(dataset,entonema,wav), './temp/{}'.format(entonema), 1)
'''

'''
xlsx = '../audios/definite/features4.xlsx'
data = utils.get_data(xlsx)
gb = data.groupby('entonema')
ct = gb.count()

print('ct:\n{}'.format(ct))
print('ct2:\n{}'.format(data.count()))
'''