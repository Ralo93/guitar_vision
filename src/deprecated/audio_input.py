import pyaudio
import wave
import os
import numpy as np

def check_microphone():
    p = pyaudio.PyAudio()
    info = p.get_host_api_info_by_index(0)
    num_devices = info.get('deviceCount')
    
    # Check if there's a microphone input device
    for i in range(0, num_devices):
        if p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels') > 0:
            print(f"Microphone found: {p.get_device_info_by_host_api_device_index(0, i).get('name')}")
            p.terminate()
            return True
    p.terminate()
    print("No microphone found!")
    return False

def capture_audio():
    if not check_microphone():
        return
    
    chunk = 1024  # Record in chunks of 1024 samples
    sample_format = pyaudio.paInt32  # 16 bits per sample
    channels = 1  # Mono recording
    fs = 44100  # Record at 44100 samples per second
    seconds = 4  # Duration of recording
    output_file = 'data/interim/tmp.wav'
    
    p = pyaudio.PyAudio()

    try:
        # Open the stream
        stream = p.open(format=sample_format, channels=channels,
                        rate=fs, frames_per_buffer=chunk, input=True)

        frames = []

        print("Recording...")

        # Capture audio data
        for _ in range(0, int(fs / chunk * seconds)):
            data = stream.read(chunk)
            frames.append(data)

        print("Recording finished")

        # Stop and close the stream
        stream.stop_stream()
        stream.close()

    except Exception as e:
        print(f"Error while recording: {e}")
    
    finally:
        p.terminate()

    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Save the recorded audio as a .wav file
    wf = wave.open(output_file, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(sample_format))
    wf.setframerate(fs)
    wf.writeframes(b''.join(frames))
    wf.close()

    print(f"Audio saved to {output_file}")

    #if return_data:
        # Convert audio data to NumPy array suitable for librosa
    audio_data = np.frombuffer(b''.join(frames), dtype=np.int16).astype(np.float32) / 32768.0  # Normalize the int16 data
    return audio_data

# Call the function
capture_audio()
