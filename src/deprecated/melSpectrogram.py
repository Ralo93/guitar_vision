import torch
import torchaudio
import matplotlib.pyplot as plt
from IPython.display import Audio

# Load the audio file
waveform, sample_rate = torchaudio.load(r'C:\Users\rapha\repositories\guitar_hero\data\raw\E-Minor (2).wav')

# Create the Mel spectrogram transform
transform = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_mels=64)

# Apply the transform to the waveform
mel_spectrogram = transform(waveform)

# Convert to decibels
mel_spectrogram_db = torchaudio.transforms.AmplitudeToDB()(mel_spectrogram)

# Plot the Mel spectrogram
plt.figure(figsize=(12, 8))
plt.imshow(mel_spectrogram_db[0].numpy(), aspect='auto', origin='lower', cmap='autumn')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel Spectrogram')
plt.xlabel('Time')
plt.ylabel('Mel Frequency')
plt.tight_layout()
plt.show()

# Play the audio
Audio(waveform.numpy()[0], rate=sample_rate)

# Print some information
print(f"Shape of Mel spectrogram: {mel_spectrogram.shape}")
print(f"Min value: {mel_spectrogram.min().item():.2f}, Max value: {mel_spectrogram.max().item():.2f}")