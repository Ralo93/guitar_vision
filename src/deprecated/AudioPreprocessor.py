import librosa
import numpy as np
from sklearn.preprocessing import StandardScaler

class AudioPreprocessor:
    def __init__(self, n_mfcc=13):
        self.n_mfcc = n_mfcc
        self.scaler = StandardScaler()
    
    def extract_features(self, file_path):
        # Load the audio file
        y, sr = librosa.load(file_path, sr=None)
        
        # Extract MFCCs
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)
        mfcc_mean = mfcc.mean(axis=1)
        
        # Extract Chroma Features
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = chroma.mean(axis=1)
        
        # Extract Spectral Contrast
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        contrast_mean = contrast.mean(axis=1)
        
        # Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(y)
        zcr_mean = zcr.mean()
        
        # RMS Energy
        rms = librosa.feature.rms(y=y)
        rms_mean = rms.mean()

        # Combine all features into one vector
        feature_vector = np.concatenate((mfcc_mean, chroma_mean, contrast_mean, [zcr_mean], [rms_mean]))

        # Standardize the feature vector
        feature_vector_scaled = self.scaler.fit_transform([feature_vector])

        return feature_vector_scaled
