import crepe
import torchaudio
import torch
import os

def load_audio(file_path, sample_rate=16000):
    """Load an audio file and resample it to the target sample rate."""
    waveform, sr = torchaudio.load(file_path)
    
    # Resample if necessary
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(sr, sample_rate)
        waveform = resampler(waveform)
    
    # Convert to mono if stereo
    if waveform.size(0) > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    return waveform, sample_rate

def predict_pitch(file_path):
    """Predict pitch using CREPE."""
    # Load the audio file
    waveform, sample_rate = load_audio(file_path)
    
    # Convert the waveform to a numpy array as required by CREPE
    waveform_np = waveform.squeeze().numpy()
    
    # Call CREPE's predict function
    _, frequency, confidence, _ = crepe.predict(waveform_np, sample_rate, viterbi=True)
    
    # Print the predicted frequencies and their confidence scores
    print("Predicted Frequencies (Hz):", frequency)
    print("Confidence Scores:", confidence)

if __name__ == "__main__":
    # Path to your audio file
    file_path = r"C:\Users\rapha\repositories\guitar_vision\data\raw\kaggle_chords\Testing\Am\Am_AcusticVince_JO_1-Minor.wav"  # Replace with the actual file path

    # Ensure the file exists
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
    else:
        # Call the function to predict pitch using CREPE
        predict_pitch(file_path)
