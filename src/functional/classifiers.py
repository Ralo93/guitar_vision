import torch
import torch.nn as nn
import torchaudio
import torchvggish
import os
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import torchaudio.transforms as T


class ChordClassifier(nn.Module):
    
    def __init__(self, num_classes=1):
        super().__init__()
        self.feature_extractor = torchvggish.vggish()

        # Freeze layers, but keep last layers unfrozen
        for name, param in self.feature_extractor.named_parameters():
            if 'embeddings.4' in name or 'embeddings.2' in name or 'embeddings.0' in name:  # Unfreeze only the last layer
                param.requires_grad = False # Change this as well for unfreezing!
                print(f"Unfrozen parameter: {name}, requires_grad: {param.requires_grad}")
            else:
                param.requires_grad = False  # Keep all other layers frozen
                print(f"Frozen parameter: {name}, requires_grad: {param.requires_grad}")
        
            
        # Simple classifier on top of VGGish embeddings
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),  # VGGish outputs 128-dimensional embeddings
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_classes),
            nn.Sigmoid()
        )


    def forward(self, x):
        # Remove torch.no_grad() here
        features = self.feature_extractor(x)  # Remove torch.no_grad() here
        return self.classifier(features)