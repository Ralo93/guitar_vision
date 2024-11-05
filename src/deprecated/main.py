

import os
import librosa
import soundfile as sf

def resample_to_fixed_duration(directory, target_duration=0.96, target_sr=22050):
    '''
    Resamples all audio files in a given directory to be exactly 0.96 seconds long.
    Outputs the resampled audio files back into the same directory.

    Parameters:
    - directory: str, path to the directory containing audio files
    - target_duration: float, duration (in seconds) to resample each audio file to
    - target_sr: int, target sampling rate (default 22050 Hz)
    '''
    
    for filename in os.listdir(directory):
        if filename.endswith(('.wav', '.mp3', '.flac')):  # Check for audio file types
            filepath = os.path.join(directory, filename)
            
            # Load the audio file
            y, sr = librosa.load(filepath, sr=target_sr)
            
            # Calculate the target number of samples for 0.96 seconds
            target_samples = int(target_duration * target_sr)
            
            # Adjust the audio length
            if len(y) < target_samples:
                # If audio is shorter, pad with zeros
                y = librosa.util.fix_length(y, target_samples)
            else:
                # If audio is longer, truncate
                y = y[:target_samples]
            
            # Save the resampled audio back to the directory
            output_path = os.path.join(directory, f"resampled_{filename}")
            sf.write(output_path, y, target_sr)
            print(f"Resampled file saved as: {output_path}")



def compress_to_fixed_duration(directory, target_duration=0.96, target_sr=22050):
    '''
    Compresses all audio files in a given directory to be exactly 0.96 seconds long
    without changing the pitch. Outputs the modified audio files back into the same directory.

    Parameters:
    - directory: str, path to the directory containing audio files
    - target_duration: float, duration (in seconds) to compress each audio file to
    - target_sr: int, target sampling rate (default 22050 Hz)
    '''
    
    for filename in os.listdir(directory):
        if filename.endswith(('.wav', '.mp3', '.flac')):  # Check for audio file types
            filepath = os.path.join(directory, filename)
            
            # Load the audio file
            y, sr = librosa.load(filepath, sr=target_sr)
            
            # Calculate the time-stretch factor
            original_duration = librosa.get_duration(y=y, sr=sr)
            stretch_factor = original_duration / target_duration
            
            # Apply time-stretching to compress or expand the duration
            y_stretched = librosa.effects.time_stretch(y, rate=stretch_factor)
            
            # Ensure the stretched audio matches the target length exactly
            target_samples = int(target_duration * target_sr)
            y_stretched = librosa.util.fix_length(y_stretched, size=target_samples)
            
            # Save the modified audio back to the directory
            output_path = os.path.join(directory, f"compressed_{filename}")
            sf.write(output_path, y_stretched, target_sr)
            print(f"Compressed file saved as: {output_path}")



def main():


    # Usage example:
    #resample_to_fixed_duration(r"C:\Users\rapha\repositories\guitar_vision\audio")
    # Usage example:
    compress_to_fixed_duration(r"C:\Users\rapha\repositories\guitar_vision\audio")



if __name__ == "__main__":
    


    main()
