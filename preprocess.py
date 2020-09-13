import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def preprocess_dataframe(data, feature__selection = lambda X, y: X, pca_components = 0, scaler = 0):
    y = data['entonema']
    y = pd.DataFrame(list(map(str,y.to_numpy()))).values.ravel()
    cols = data.columns.to_list()
    cols.remove('entonema')
    if 'wav' in cols: cols.remove('wav')
    if 'Unnamed: 0' in cols: cols.remove('Unnamed: 0')
    if 'U' in cols: cols.remove('U')
    if 'Unnamed: 0.1' in cols: cols.remove('Unnamed: 0.1')
    X = data.loc[:, cols]


    std_scaler = StandardScaler()
    
    if scaler == 0:
        X = std_scaler.fit_transform(X)
    else:
        X = scaler.transform(X)
        std_scaler = scaler

    return X, y, cols, std_scaler