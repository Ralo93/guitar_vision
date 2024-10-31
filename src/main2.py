# Initialize the preprocessor and classifier
import os
import numpy as np

from AudioPreprocessor import *
from ChordClassifier import *
from RealTimeChordClassifier import *
from AudioFeatureVisualizer import *

base_path = r'C:\Users\rapha\repositories\guitar_hero\data\raw'

preprocessor = AudioPreprocessor()
classifier_yamnet = ChordClassifier(model_type='random_forest', transfer_model='vggish')

# Load your dataset of audio files and labels (this is a simplified example)
audio_files = [f for f in os.listdir(base_path) if os.path.isfile(os.path.join(base_path, f))]

labels = [os.path.splitext(f)[0].replace(' (2)', '') for f in audio_files]

# Convert back to list for further processing
y = list(labels)

# Preprocess each audio file to extract features
features = []
for file in audio_files:
    features.append(preprocessor.extract_features(os.path.join(base_path, file)))
    
# Replace 'your_audio_file.wav' with the path to your audio file
file_path = os.path.join(base_path, file)
print(len(features))
#visualizer = AudioFeatureVisualizer(file_path)
#visualizer.plot_all_features()

#print(features)

# Flatten the list into a proper feature matrix
X = np.vstack(features)

# Train the model
classifier_yamnet.train(X, y)

# Now, let's evaluate a new recording in real-time
real_time_evaluator = RealTimeChordClassifier(preprocessor, classifier_yamnet)


chord_prediction = real_time_evaluator.evaluate_live_recording(r'C:\Users\rapha\repositories\guitar_hero\data\interim\tmp.wav')

print(f"The predicted chord is: {chord_prediction}")



