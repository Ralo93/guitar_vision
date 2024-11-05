class RealTimeChordClassifier:
    def __init__(self, preprocessor, classifier):
        self.preprocessor = preprocessor
        self.classifier = classifier
    
    def evaluate_live_recording(self, file_path):
        # Preprocess the audio file
        features = self.preprocessor.extract_features(file_path)
        
        # Predict the chord
        prediction = self.classifier.predict(features)
        
        return prediction
