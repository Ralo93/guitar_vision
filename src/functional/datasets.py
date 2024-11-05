import os
import torch
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torchaudio
import torchvggish
import os
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import torchaudio.transforms as T


class ChordDataset(Dataset):
    
    def __init__(self, file_paths, labels, sample_rate=16000):
        self.file_paths = file_paths
        self.labels = labels
        self.sample_rate = sample_rate
        
        # VGGish expects 96 mel bands
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=2048,
            win_length=400,
            hop_length=160,
            n_mels=96,
            f_min=125,
            f_max=7500
        )
        
        # Log mel spectrogram
        self.amplitude_to_db = T.AmplitudeToDB()
        
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        # Load audio
        waveform, sr = torchaudio.load(self.file_paths[idx])
        #print(f"Getting: {self.file_paths[idx]}")
        
        # Resample if necessary
        if sr != self.sample_rate:
            resampler = T.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        
        # Convert to mono if stereo
        if waveform.size(0) > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Get mel spectrogram
        mel_spec = self.mel_transform(waveform)
        
        # Convert to dB scale
        mel_spec = self.amplitude_to_db(mel_spec)

        # Add normalization step for mel spectrograms
        mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-8)
        #print(f"Getting mel_spec: {mel_spec}")
        
        # VGGish expects input size of (batch_size, 1, 96, 64)
        # So we need to ensure our time dimension is 64 frames
        target_length = 64
        current_length = mel_spec.size(2)
        
        if current_length < target_length:
            # Pad if too short
            padding = target_length - current_length
            mel_spec = torch.nn.functional.pad(mel_spec, (0, padding))
            
        elif current_length > target_length:
            # Take center portion if too long
            start = (current_length - target_length) // 2
            mel_spec = mel_spec[:, :, start:start + target_length]
        
        # Add channel dimension
       # mel_spec = mel_spec.unsqueeze(0)
            
        return mel_spec, torch.tensor(self.labels[idx], dtype=torch.float32)