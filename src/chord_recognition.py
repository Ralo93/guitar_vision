import librosa
import numpy as np


# Basic major and minor triad intervals
major_intervals = [0, 4, 7]  # Root, major third, perfect fifth
minor_intervals = [0, 3, 7]  # Root, minor third, perfect fifth

# Notes for chroma mapping
note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

def recognize_chord(audio_data, sr=44100):
    # Extract chroma features
    chroma = librosa.feature.chroma_stft(y=audio_data, sr=sr)
    # Average the chroma vectors across frames to get a global representation
    chroma_mean = np.mean(chroma, axis=1)

    print(chroma_mean)
    
    # Find the index of the root note (the note with the highest chroma value)
    root_index = np.argmax(chroma_mean)
    root_note = note_names[root_index]
    
    # Get the chroma values relative to the root
    shifted_chroma = np.roll(chroma_mean, -root_index)
    
    # Check for major or minor chord by comparing the chroma pattern
    if np.allclose(shifted_chroma[major_intervals], [shifted_chroma[0], shifted_chroma[4], shifted_chroma[7]], atol=0.2):
        return f"{root_note}"
    elif np.allclose(shifted_chroma[minor_intervals], [shifted_chroma[0], shifted_chroma[3], shifted_chroma[7]], atol=0.2):
        return f"{root_note}m"
    else:
        return f"{root_note} chord (type undetermined)"
