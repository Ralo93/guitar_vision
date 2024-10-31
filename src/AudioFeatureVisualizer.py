import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

class AudioFeatureVisualizer:
    def __init__(self, file_path, n_mfcc=13):
        self.file_path = file_path
        self.n_mfcc = n_mfcc
        self.y, self.sr = librosa.load(self.file_path, sr=None)
        self.extract_features()

        '''
        Visual Analogy
        Imagine each feature as a different aspect of a painting:

        MFCCs are like the colors used.
        Chroma Features are the specific shapes or objects in the painting.
        Spectral Contrast highlights the light and shadow areas.
        Zero Crossing Rate adds the texture and brush strokes.
        RMS Energy adjusts the overall brightness or darkness.
        '''
    
    def extract_features(self):
        # Extract MFCCs
        self.mfcc = librosa.feature.mfcc(y=self.y, sr=self.sr, n_mfcc=self.n_mfcc)
        self.mfcc_mean = self.mfcc.mean(axis=1)
        
        # Extract Chroma Features
        self.chroma = librosa.feature.chroma_stft(y=self.y, sr=self.sr)
        self.chroma_mean = self.chroma.mean(axis=1)
        
        # Extract Spectral Contrast
        self.contrast = librosa.feature.spectral_contrast(y=self.y, sr=self.sr)
        self.contrast_mean = self.contrast.mean(axis=1)
        
        # Zero Crossing Rate
        self.zcr = librosa.feature.zero_crossing_rate(y=self.y)
        self.zcr_mean = self.zcr.mean()
        
        # RMS Energy
        self.rms = librosa.feature.rms(y=self.y)
        self.rms_mean = self.rms.mean()
    
    def plot_mfcc(self):
        '''
        This gets the timbre or color of a sound. Difference between guitar and piano.
        Mel Frequency Cepstral Coefficients
        '''
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(self.mfcc, x_axis='time', sr=self.sr, y_axis='mel')
        plt.colorbar()
        plt.title('MFCC')
        plt.ylabel('MFCC Coefficients')  # Added y-axis label
        plt.tight_layout()
        plt.show()
    
    def plot_chroma(self):
        '''
        This gets the pitch of the sound. e.g., C#
        Chroma Features
        '''
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(self.chroma, x_axis='time', sr=self.sr, cmap='autumn', y_axis='chroma')
        plt.colorbar()
        plt.title('Chroma Features')
        plt.ylabel('Pitch Classes')  # Added y-axis label
        plt.tight_layout()
        plt.show()
    
    def plot_spectral_contrast(self):
        '''
        This gets the difference in loudness between peaks and valleys in the sound's frequency spectrum.
        It highlights the high and low points.
        '''
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(self.contrast, x_axis='time', sr=self.sr, cmap='autumn', y_axis='linear')
        plt.colorbar()
        plt.title('Spectral Contrast')
        plt.ylabel('Frequency Bands (Hz)')  # Added y-axis label
        plt.tight_layout()
        plt.show()
    
    def plot_zcr(self):
        '''
        This measures how frequently the signal crosses the zero amplitude line.
        Zero Crossing Rate
        '''
        plt.figure(figsize=(10, 4))
        plt.plot(self.zcr[0], label='Zero Crossing Rate')
        plt.xlabel('Frames')
        plt.ylabel('ZCR')
        plt.title('Zero Crossing Rate')
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    def plot_rms(self):
        '''
        This measures the power or loudness of the audio signal.
        RMS Energy
        '''
        plt.figure(figsize=(10, 4))
        plt.plot(self.rms[0], label='RMS Energy', color='orange')
        plt.xlabel('Frames')
        plt.ylabel('RMS')
        plt.title('RMS Energy')
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    def plot_all_features(self):
        self.plot_mfcc()
        self.plot_chroma()
        self.plot_spectral_contrast()
        #self.plot_zcr()
        self.plot_rms()

# Usage Example
if __name__ == "__main__":

    file_path = r"C:\Users\rapha\repositories\guitar_vision\data\raw\A-Major.wav"
    
    visualizer = AudioFeatureVisualizer(file_path)
    visualizer.plot_all_features()
