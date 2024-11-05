from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import torch
import tensorflow_hub as hub
import librosa
import numpy as np

class ChordClassifier:
    def __init__(self, model_type='svm', transfer_model=None):
        self.transfer_model = transfer_model
        if model_type == 'svm':
            self.model = SVC(kernel='linear')
        elif model_type == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=300)
        elif model_type == 'knn':
            self.model = KNeighborsClassifier(n_neighbors=5)
        elif model_type == 'logistic_regression':
            self.model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=200)
        else:
            raise ValueError("Model type not supported. Choose 'svm', 'random_forest', 'knn', or 'logistic_regression'.")

    def extract_features(self, audio_file):
        # Load and preprocess audio data
        audio, sr = librosa.load(audio_file, sr=16000)
        
        if self.transfer_model == 'vggish':
            # Extract VGGish features
            # Placeholder function, assuming you already have the pre-trained model.
            return self.extract_vggish_features(audio)
        elif self.transfer_model == 'yamnet':
            # Extract YAMNet features
            return self.extract_yamnet_features(audio)
        elif self.transfer_model == 'openl3':
            # Extract OpenL3 features
            return self.extract_openl3_features(audio)
        else:
            raise ValueError("Transfer model not supported. Choose 'vggish', 'yamnet', or 'openl3'.")

    def extract_vggish_features(self, audio):
        # Load VGGish from TensorFlow Hub and preprocess audio
        model = hub.load('https://tfhub.dev/google/vggish/1')
        # Process your audio here using the VGGish feature extractor
        # Placeholder: return extracted features
        return np.zeros((1, 128))  # Dummy output, replace with actual features
    
    def extract_yamnet_features(self, audio):
        # Load YAMNet from TensorFlow Hub and preprocess audio
        yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
        embeddings, _, _ = yamnet_model(audio)
        return embeddings.numpy()

    #def extract_openl3_features(self, audio):
    #    import openl3
        # Extract OpenL3 features
    #    emb, ts = openl3.get_audio_embedding(audio, sr=16000, content_type='music')
    #    return emb

    def train(self, X, y):
        # Split into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the model
        self.model.fit(X_train, y_train)

        # Evaluate the model on the test set
        test_accuracy = self.model.score(X_test, y_test)
        print(f"Test Accuracy: {test_accuracy:.2f}")

        # Evaluate the model on the train set
        train_accuracy = self.model.score(X_train, y_train)
        print(f"Train Accuracy: {train_accuracy:.2f}")

        return test_accuracy

    def predict(self, audio_file):
        # Extract features using the selected transfer learning model
        features = self.extract_features(audio_file)
        
        # Predict using the trained model
        return self.model.predict(features)
