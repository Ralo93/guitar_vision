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
        Imagine each feature as a different aspect of a painting.
        MFCCs are like the colors used.
        Chroma Features are the specific shapes or objects in the painting.
        Spectral Contrast highlights the light and shadow areas.
        Zero Crossing Rate adds the texture and brush strokes.
        RMS Energy adjusts the overall brightness or darkness.
        '''

    def plot_fft(self):
        '''
        This function performs a Fast Fourier Transform (FFT) on the audio signal to
        visualize its frequency spectrum. It shows the frequencies present in the sound
        and their respective amplitudes.
        '''
        # Perform FFT
        fft = np.fft.fft(self.y)
        magnitude = np.abs(fft)  # Get the magnitude of each frequency component
        frequency = np.linspace(0, self.sr, len(magnitude))  # Frequency axis

        # Plot FFT
        plt.figure(figsize=(10, 4))
        plt.plot(frequency[:len(frequency)//2], magnitude[:len(magnitude)//2])  # Only plot the positive frequencies
        plt.title('FFT of Audio Signal')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.tight_layout()
        plt.show()



    def extract_features(self):
        # Extract MFCCs
        self.mfcc = librosa.feature.mfcc(y=self.y, sr=self.sr, n_mfcc=self.n_mfcc)
        self.mfcc_mean = self.mfcc.mean(axis=1)
        
        # Extract Chroma Features
        self.chroma = librosa.feature.chroma_stft(y=self.y, sr=self.sr)
        print(self.chroma.shape)

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
        librosa.display.specshow(self.mfcc, x_axis='time', sr=self.sr, y_axis='mel', cmap='magma')
        plt.colorbar()
        plt.title('MFCC')
        plt.ylabel('MFCC Coefficients')  # Added y-axis label
        plt.tight_layout()
        plt.show()

    def mfcc(self):
        # Use instance variables for audio data and sample rate
        mel_spect = librosa.feature.melspectrogram(y=self.y, sr=self.sr, n_fft=2048, hop_length=1024)
        
        # Convert to decibel scale for better visualization
        mel_spect_db = librosa.power_to_db(mel_spect, ref=np.max)
        
        # Plot the Mel spectrogram
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(mel_spect_db, y_axis='mel', fmax=8000, x_axis='time', cmap='magma')
        plt.title('Mel Spectrogram')
        plt.colorbar(format='%+2.0f dB')
        plt.tight_layout()
        plt.show()

    
    def plot_chroma(self):
        '''
        This gets the pitch of the sound. e.g., C#
        Chroma Features
        '''
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(self.chroma, x_axis='time', sr=self.sr, cmap='magma', y_axis='chroma')
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
        librosa.display.specshow(self.contrast, x_axis='time', sr=self.sr, y_axis='linear', cmap='magma')
        plt.colorbar()
        plt.title('Spectral Contrast')
        plt.ylabel('Frequency Bands (Hz)')  # Added y-axis label
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
    
    

    def plot_waveform(self):
        '''
        This function plots the original audio signal in the time domain.
        It shows how the amplitude of the signal varies over time.
        '''
        # Time axis for the signal
        time = np.linspace(0, len(self.y) / self.sr, num=len(self.y))
        
        # Plot waveform
        plt.figure(figsize=(10, 4))
        plt.plot(time, self.y, label='Amplitude')
        plt.title('Audio Signal Waveform')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_spectrogram(self):
        '''
        This function computes and plots a regular spectrogram from the audio signal using STFT.
        It shows the frequency content of the signal over time without mapping to the Mel scale.
        '''
        # Compute the Short-Time Fourier Transform (STFT)
        stft = np.abs(librosa.stft(self.y))
        
        # Convert amplitude to decibel scale for better visualization
        stft_db = librosa.amplitude_to_db(stft, ref=np.max)
        
        # Plot the spectrogram (regular frequency scale)
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(stft_db, sr=self.sr, x_axis='time', y_axis='log')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Spectrogram')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Frequency (Hz)')
        plt.tight_layout()
        plt.show()

    def plot_mel_spectrogram(self):
        '''
        This function computes and plots a Mel spectrogram from the audio signal.
        It applies the Mel filter, which maps frequencies to the Mel scale for a more perceptual representation.
        '''
        # Compute the Mel-scaled spectrogram
        mel_spect = librosa.feature.melspectrogram(y=self.y, sr=self.sr, n_fft=2048, hop_length=1024, n_mels=40)
        
        # Convert the Mel spectrogram to decibel scale
        mel_spect_db = librosa.power_to_db(mel_spect, ref=np.max)
        
        # Plot the Mel spectrogram
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(mel_spect_db, sr=self.sr, x_axis='time', y_axis='mel', cmap='magma', fmax=8000)
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel Spectrogram')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Mel Frequency (Hz)')
        plt.tight_layout()
        plt.show()



    
    def plot_all_features(self):
        #self.plot_spectrogram()
        #self.plot_mel_spectrogram()
        self.plot_chroma()
        #self.plot_spectral_contrast()
        #self.plot_zcr()
        #self.plot_rms()
        #self.plot_spectrogram()
        #self.plot_fft()

# Usage Example
if __name__ == "__main__":

    file_path = r"C:\Users\rapha\repositories\guitar_vision\data\raw\A-Major.wav"
    
    visualizer = AudioFeatureVisualizer(file_path)
    visualizer.plot_all_features()
