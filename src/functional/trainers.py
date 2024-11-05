from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torchaudio
import torchvggish
import os
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import torchaudio.transforms as T


class ChordTrainer:
    def __init__(self, model, criterion, optimizer, device, save_path=r'src\Neuer Ordner\models\best_model2.pth', threshold=0.5):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.save_path = save_path  # Path where the model checkpoints will be saved
        self.best_val_loss = float('inf')  # Initialize with a high value
        self.threshold = threshold
        self.train_losses = []
        self.val_losses = []
        
    def train_epoch(self, train_dataloader, val_dataloader):
        self.model.train()
        running_loss = 0.0
        epoch_losses = []  # Store all batch losses for this epoch

        for i, (inputs, labels) in enumerate(train_dataloader):
            print("Processing batch", i + 1)
            
            inputs, labels = inputs.to(self.device), labels.to(self.device)
              
            outputs = self.model(inputs)
            loss = self.criterion(outputs.squeeze(), labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()           
            running_loss += loss.item()
            
            # Every 3 batches, compute average training loss and validation loss
            if (i + 1) % 3 == 0:
                avg_train_loss = running_loss / 3
                epoch_losses.append(avg_train_loss)
                print(f'Batch {i + 1}, Training Loss: {avg_train_loss:.4f}')
                running_loss = 0.0

                # Compute validation loss
                val_loss = self.evaluate_loss(val_dataloader)
                print(f'Batch {i + 1}, Validation Loss: {val_loss:.4f}')

                # Save the model if validation loss improves
                self.save_checkpoint(val_loss)

        # Store average losses for the epoch
        self.train_losses.append(sum(epoch_losses) / len(epoch_losses))
        self.val_losses.append(val_loss)  # Store the last validation loss of the epoch


    def evaluate_loss(self, dataloader):
        self.model.eval()
        running_loss = 0.0
        with torch.no_grad(): #forward pass, no need to store the gradients
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs.squeeze(), labels)
                running_loss += loss.item()

        # Return average loss over the validation set
        return running_loss / len(dataloader)

    def evaluate(self, dataloader):
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                predicted = (outputs.squeeze() > self.threshold).float()
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        return classification_report(
            all_labels, 
            all_preds, 
            target_names=["Minor", "Major"], 
            zero_division=0
        )
    

    def save_checkpoint(self, val_loss):
        """Saves the model if the validation loss improves."""
        if val_loss < self.best_val_loss:
            print(f'Validation loss improved from {self.best_val_loss:.4f} to {val_loss:.4f}. Saving model...')
            torch.save(self.model.state_dict(), self.save_path)
            self.best_val_loss = val_loss
        else:
            print(f'Validation loss did not improve (Best: {self.best_val_loss:.4f}).')

    def plot_losses(self):
        """Plot training and validation losses."""
        plt.figure(figsize=(10, 6))
        epochs = range(1, len(self.train_losses) + 1)
        
        plt.plot(epochs, self.train_losses, 'b-', label='Training Loss')
        plt.plot(epochs, self.val_losses, 'r-', label='Validation Loss')
        
        plt.title('Training and Validation Loss Over Time')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Add markers to show minimum validation loss
        min_val_loss = min(self.val_losses)
        min_val_epoch = self.val_losses.index(min_val_loss) + 1
        plt.plot(min_val_epoch, min_val_loss, 'go', label=f'Best Validation Loss: {min_val_loss:.4f}')
        
        plt.legend()
        plt.tight_layout()
        
        # Save the plot
        plt.savefig('training_history.png')
        plt.show()
            