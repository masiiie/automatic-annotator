import sys, os
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *


from PyQt5.QtMultimedia import QMediaPlayer
from PyQt5 import QtMultimedia


from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from scipy.io import wavfile
from scipy import signal
from skimage.color import rgb2grey

from numpy.fft import fftshift

from joblib import dump, load
import pickle
from statistics import naive_vectorizer
from sklearn.preprocessing import StandardScaler
from comparing_pitches import get_pitch_decompy_values


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)

        self.axes = self.fig.add_subplot(111)
        self.axes.set_title('Espectrograma')
        self.axes.set_ylabel('Frequency [Hz]')
        self.axes.set_xlabel('Time [sec]')


        super(MplCanvas, self).__init__(self.fig)


        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                QSizePolicy.Expanding,
                QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)


    def plot_spectrogram(self, path):
        self.axes.clear()
        self.axes.set_title('Espectrograma')
        sample_rate, samples = wavfile.read(path)
        
        nfft = 1024
        Pxx, freqs, bins, im = self.axes.specgram(samples, NFFT=nfft, Fs=sample_rate, noverlap=900)

        '''
        frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate, return_onesided=False)   
        self.axes.pcolormesh(times, fftshift(frequencies), fftshift(spectrogram, axes=0), shading='gouraud')
        name = os.path.basename(path)
        '''
        
        self.draw()
        



class Window(QMainWindow):
    
    def __init__(self):
        QMainWindow.__init__(self)
        self.resize(600, 650)
        self.setWindowTitle("Anotador prosódico.")       
        self.player = QMediaPlayer(None)
        
        self.slider = QSlider() 
        self.slider.setGeometry(10, 10, 380, 25)
        
        self.play_button = QPushButton(self)
        self.play_button.setText("Reproducir")
        self.play_button.setGeometry(10, 70, 100, 25)
        self.play_button.setObjectName("play_button")
        
        self.pause_button = QPushButton(self)
        self.pause_button.setText("Pausar")
        self.pause_button.setGeometry(120, 70, 100, 25)
        self.pause_button.setObjectName("pause_button")
        
        self.stop_button = QPushButton(self)
        self.stop_button.setText("Detener")
        self.stop_button.setGeometry(230, 70, 100, 25)
        self.stop_button.setObjectName("stop_button")


        
        self.track_label = QLabel(self)
        self.track_label.setText(
            u"No se ha seleccionado ningún archivo."
        )
        self.track_label.setGeometry(10, 20, 340, 25)
        self.track_label.setObjectName("track_label")
        
        self.browse_button = QPushButton(self)
        self.browse_button.setText("Seleccionar archivo")
        self.browse_button.setGeometry(355, 20, 100, 25)
        self.browse_button.setObjectName("browse_button")
        


        self.canvas = MplCanvas(self, width=5, height=4, dpi=100)
        self.canvas.setGeometry(50, 120, 500, 400)


        self.entonema_label = QLabel(self)
        self.entonema_label.setText(
            u"Entonema: Desconocido"
        )
        self.entonema_label.setGeometry(10, 570, 340, 50)
        self.entonema_label.setObjectName("entonema_label")

        

        #self.clf = load('./modelos entrenados/svm_experimento2.joblib') 
        with open('./modelos entrenados/SVC.pkl', 'rb') as f:
            self.clf, self.scaler, self.cols = pickle.load(f)

        
        QMetaObject.connectSlotsByName(self)
    
    def on_play_button_pressed(self):
        self.player.play()


    def on_pause_button_pressed(self):
        self.player.pause()
    
    def on_stop_button_pressed(self):
        self.player.stop()
    
    def on_browse_button_released(self):
        #path = unicode(QFileDialog.getOpenFileName(self))
        path = QFileDialog.getOpenFileName(self)[0]

        name = os.path.basename(path)
        if not name.endswith('.wav'):
            self.track_label.setText(u"{} no es un archivo .wav".format(name))
            return
            

        self.player.setMedia(QtMultimedia.QMediaContent(QUrl.fromLocalFile(path)))
        self.track_label.setText(name)
        self.canvas.plot_spectrogram(path)

        print('path = {}'.format(path))
        curve = get_pitch_decompy_values(path, remove_silencess = True, interpolate = False)
        vector, _ = naive_vectorizer(curve, true_features = self.cols)
        X = [vector]
        X = self.scaler.transform(X)
        predicted = self.clf.predict(X)[0]

        print('predicted = {}'.format(predicted))


        self.entonema_label.setText(
            u"Predicción de entonema: {}".format(predicted)
        )




if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec_())