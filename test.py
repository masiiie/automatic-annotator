from statistics import naive_vectorizer
import pickle, os



with open('./modelos entrenados/SVC.pkl', 'rb') as f:
    modelo = pickle.load(f)



#for ent in os.listdir('../II/audios/definite'):
ent = '7'
for wav in os.listdir('../II/audios/definite/{}'.format(ent)):
    print('wav: {}  ent: {}'.format(wav, ent))
    vector = naive_vectorizer('../II/audios/definite/{}/{}'.format(ent, wav))
    predicted = modelo.predict([vector])[0]
    print('predicted = {}\n\n'.format(predicted))
