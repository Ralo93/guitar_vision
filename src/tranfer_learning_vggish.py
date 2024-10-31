import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
import tensorflow_hub as hub
import tensorflow as tf

class SmallDataset(Dataset):
    """A simple dataset for demonstration purposes."""
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample, label

def load_model_from_hub(url, save_path):
    """
    Load a model from TensorFlow Hub, save it locally, and return a PyTorch model.
    """
    # Load the TensorFlow Hub model
    model = hub.load(url)
    
    # Save the model locally
    tf.saved_model.save(model, save_path)
    print(f"Model saved to {save_path}")

    return model

def inspect_model(model):
    """
    Inspect the model's architecture and parameters.
    """
    model.summary()  # This works for TensorFlow models
    for layer in model.layers:
        print(f"Layer: {layer.name}, Output shape: {layer.output_shape}, Trainable: {layer.trainable}")

def convert_to_pytorch(model):
    """
    Extract the core model architecture from TensorFlow Hub and use it in PyTorch.
    """
    # Extract the feature layers from the TF model (remove final classification layer)
    feature_extractor = model.signatures['serving_default']
    
    # Assuming we get embeddings as output, this would be the feature extractor
    return feature_extractor

class TransferLearningModel(nn.Module):
    """
    Custom PyTorch model for transfer learning.
    """
    def __init__(self, feature_extractor, num_classes=13):
        super(TransferLearningModel, self).__init__()
        self.feature_extractor = feature_extractor  # Pre-trained feature extractor
        self.fc = nn.Linear(1280, num_classes)  # Adjust this based on the output size of your extractor

    def forward(self, x):
        with torch.no_grad():
            # Extract features from the pre-trained model
            features = self.feature_extractor(x)
        # Pass through the new classification layer
        x = self.fc(features)
        return x

def freeze_parameters(model):
    """
    Freeze all parameters in the feature extractor and only keep the final layer trainable.
    """
    # Freeze all layers except the final fully connected layer
    for param in model.feature_extractor.parameters():
        param.requires_grad = False
    print("All parameters frozen except the final layer.")

def train_model(model, dataloader, num_epochs=5, learning_rate=0.001):
    """
    Train the modified model on a very small dataset.
    """
    # Use cross-entropy loss for classification
    criterion = nn.CrossEntropyLoss()
    
    # Only optimize the final layer
    optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)
    
    # Training loop
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in dataloader:
            # Move inputs and labels to GPU if available
            inputs, labels = inputs.cuda(), labels.cuda()
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss:.4f}")
    
    print("Training complete.")

def prepare_dataloader(data, labels, batch_size=4):
    """
    Prepare a DataLoader for the small dataset.
    """
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = SmallDataset(data, labels, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


# Load the pre-trained TensorFlow model from hub
tf_model = load_model_from_hub('https://tfhub.dev/google/vggish/1', save_path='./vggish_model')

# Convert to PyTorch compatible model
pytorch_model = convert_to_pytorch(tf_model)

# Create a PyTorch transfer learning model
model = TransferLearningModel(pytorch_model, num_classes=13)

# Freeze the feature extractor layers
freeze_parameters(model)

# Prepare a small dataset for training
data = [...]  # Your data here (could be images, spectrograms, etc.)
labels = [...]  # Corresponding labels
dataloader = prepare_dataloader(data, labels, batch_size=4)

# Train the model
train_model(model, dataloader, num_epochs=5)
