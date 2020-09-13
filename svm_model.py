from make_panda_set import get_data
from sklearn.model_selection import\
    learning_curve, ShuffleSplit, KFold, cross_val_score, RepeatedKFold, train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import time, os, utils
import matplotlib.pyplot as plt
import numpy as np
from preprocess import preprocess_dataframe
from sklearn.feature_selection import SelectKBest

from joblib import dump, load
import pickle



def definite_score(train, test, clf, pca_components = 0): 
    #input('train_columns:\n\n{}'.format(train.columns))  
    #input('test_columns:\n\n{}'.format(test.columns))  
    X_train, y_train, cols, scaler = preprocess_dataframe(train, pca_components = 0)
    X_test, y_test, _, _ = preprocess_dataframe(test, pca_components = 0, scaler = scaler)
    
    clf_name = clf.__class__.__name__



    clf.fit(X_train, y_train)

    print('El entrenamiento va a comenzar! :)')
    with open('./modelos entrenados/{}.pkl'.format(clf_name), 'wb') as f:
        # Write the model to a file.
        pickle.dump((clf, scaler, cols), f)
        print('Ya entrenamos! :)')
        #return

    y_pred = clf.predict(X_test)


    accuracy = accuracy_score(y_test, y_pred)
    #precision = sm.precision_score(y_test, y_pred)
    #f1 = sm.f1_score(y_test, y_pred)
    print('\n\nDefinite score!')
    print('clf: {}'.format(clf.__class__.__name__))
    print('train_len = {}   test_len = {}'.format(len(train), len(test)))
    print('final_accuracy = {}\n'.format(accuracy))
    print(classification_report(y_test, y_pred))
    
    #clss = [v[0] for v in train.groupby('entonema')]
    clss = ['1','1a','1b','1c','2','2a','3','3a','3b','4','4a','5','5a','5b','6','6a','7','7a']
    title = 'Matriz de confusión\nestimador={}'.format(clf.__class__.__name__)
    plot_confusion_matrix(y_test, y_pred, clss, '', title=title)
    return accuracy

def test():
    with open('./modelos entrenados/SVC.pkl', 'rb') as f:
        clf = pickle.load(f)

    with open('./modelos entrenados/StandardScaler_SVC.pkl', 'rb') as f:
        scaler = pickle.load(f)

    naive = utils.get_data('./naive_features.xlsx')
    #naive = utils.get_data('./real.xlsx')
    X_test, y_test, _, _ = preprocess_dataframe(naive, scaler = scaler)
    y_pred = clf.predict(X_test)
        

# from sklearn.metrics import classification_report
# score = clf.score(X,y)
def svm_work(xlsx, clf, feature__selection, cv, pca_components = 0):
    start = time.time()
    data = get_data(xlsx) if type(xlsx) == type('hola') else xlsx
    X, y, cols_features = preprocess_dataframe(data, feature__selection, pca_components = pca_components)
    nsplits = int(100/40) 
    #cv = KFold(shuffle=False, n_splits=nsplits, random_state=None)

    '''
    skb = SelectKBest(k = 5)
    D = skb.fit(X, y)
    mask = skb.get_support()
    print('mask_len = {}    cols_len = {}'.format(len(mask), len(cols_features)))
    selected = [cols_features[i] for i in range(0,len(mask)) if mask[i]]
    print('5 best:  {}'.format(selected))
    input(':)')
    return
    '''

    '''
    clss = [v[0] for v in data.groupby('entonema')]
    for i in range(5):
        X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=0.40, random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print(classification_report(y_test, y_pred))
        plot_confusion_matrix(y_test,y_pred, clss, 'Confusion matrix {}'.format(i))

    input("Hoola!")
    '''

    scores = cross_val_score(clf, X, y, cv=cv, n_jobs=-1, verbose = 1)
    print('clf: {}'.format(clf))
    print('cv: {}'.format(cv))
    print('feature_selection: {}'.format(feature__selection))
    print('scores = {}'.format(scores))
    print("Accuracy: {}    Variance: {}".format(scores.mean(), scores.std() * 2))
    print('demora = {} mins'.format((time.time() - start)/60))

    
    start = time.time()
    curve_title = 'dataset_len = {} features_len = {} estimator = {}\ncv = {}\n' \
    .format(len(X), len(X[0]), clf.__class__.__name__, cv)
    plot_learning_curve(clf, curve_title, X, y, cv = cv, n_jobs=-1)
    print('demora = {} mins\n\n'.format((time.time() - start)/60))
    

def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Tamaño del conjunto de entrenamiento")
    axes[0].set_ylabel("Medida accuracy")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, return_times=True, scoring = 'accuracy', verbose=1)


    print('estimator = {0} '.format(estimator))
    print('X_len = {0}'.format(len(X)))
    print('train_size = {0} \n'.format(train_sizes))
    #print('train_scores = {0} \n'.format(train_scores))
    #print('validation_scores = {0} \n'.format(test_scores))
    #print('fit_times = {0} \n'.format(fit_times))


    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    print('Mean training scores', train_scores_mean)
    print('Mean validation scores', test_scores_mean)
    print('training_score = {}   validation_score = {}'.format(train_scores_mean.mean(), test_scores_mean.mean()))

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Conjunto de entrenamiento")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Conjunto de validación")
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-')
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    plt.savefig('chart.png')
    plt.show()
    return plt


def plot_confusion_matrix(y_true, y_pred, classes, name,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    #classes = y_true #classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='Etiqueta real',
           xlabel='Predicción')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()

    plt.savefig('./confusion_matrix_{}.png'.format(name))
    plt.show()