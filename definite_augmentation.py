from utils import make_to_given_dataset
from demo import transform
import os

def apply_transform(dataset, entonema, wav, iteations):
    file_path = '{}/{}/{}'.format(dataset, entonema, wav)
    output = './definite augmentation/{}'.format(entonema)
    if(not os.path.exists(output)):
        os.mkdir(output)
    wav = wav.replace('.wav','')
    '''
    output = '{}/{}'.format(output, wav)
    if(not os.path.exists(output)):
        os.mkdir(output)
    '''
    transform(file_path, output, iteations)



dataset = '../audios/definite'
make_to_given_dataset(dataset, apply_transform, args = [5], max_audio_by_tonema = 5)