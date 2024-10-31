
from chord_visualizer import visualize_chord
from harmonic_recommender import recommend_next_chords
from audio_input import capture_audio
from chord_recognition import recognize_chord

def main():
    #while True:
    # Capture real-time audio
    audio_data = capture_audio()
    
    # Identify the chord
    chord = recognize_chord(audio_data)

    print(chord)
    
    # Visualize the chord
    #visualize_chord(chord)
    
    # Recommend next chords based on harmonic theory
    next_chords = recommend_next_chords(chord)
    print(f"Chord: {chord}, Recommended next chords: {next_chords}")

if __name__ == "__main__":
    
    
    # for guitar: output layers are  number of chords I want to detect.

    # There should always be a non-linear activation function between layers! Otherwise they are wasted.

    # You want to compress data.

    main()
