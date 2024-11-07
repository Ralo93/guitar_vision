import os
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from utils import *


file_dir_train = r'chromas\small\training'
file_dir_test = r'chromas\small\testing'

# Gather all file paths from the directory
file_paths_train = gather_file_paths(file_dir_train)
file_paths_test = gather_file_paths(file_dir_test)


labels = [extract_label(file_path) for file_path in file_paths_train]
test_labels = [extract_label(file_path) for file_path in file_paths_test]


# Gather chroma images and labels for train and test sets
x_train = load_chroma_images(file_paths_train)
x_test = load_chroma_images(file_paths_test)

# Convert labels to numerical format if necessary (e.g., using a label encoder)
#y_train = np.array(labels)  # Assuming labels are already numerical
#y_test = np.array(test_labels)

# Encode the labels
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(labels)
y_test = label_encoder.transform(test_labels)

# Convert to TensorFlow datasets and batch
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(1000).batch(64)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(64)

print(test_ds)

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) #sparse does the one-hot-encoding on the fly, from logits: apply the softmax operation inside on the logits
optimizer = tf.keras.optimizers.Adam()

from models import *

# for plotting in tensorboard
train_loss = tf.keras.metrics.Mean() #will take the mean from the 32-dim vector
test_loss = tf.keras.metrics.Mean()
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

EPOCHS = 20
writer = tf.summary.create_file_writer('summary/other') #important! so each run will be its own graph

model = SimpleCNN(loss_object, optimizer=optimizer)

model.compile()
print(model.summary())

train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []

for epoch in range(EPOCHS):

    print(f"EPOCH: {epoch}")
    #reset the metrics
    train_loss.reset_state()
    train_accuracy.reset_state()
    test_loss.reset_state()
    test_accuracy.reset_state()
  
    for train_images, train_labels in train_ds:
        pred_train, loss_train, grads_train = model.train_step(train_images, train_labels)
        train_loss(loss_train)
        train_accuracy(train_labels, pred_train)

    for test_images, test_labels in test_ds:
        #print("testing quickly")
        pred_test, loss_test = model.test_step(test_images, test_labels)
        test_loss(loss_test)
        test_accuracy(test_labels, pred_test)


    with writer.as_default():
        tf.summary.scalar('Train Loss', train_loss.result(), step=epoch)
        tf.summary.scalar('Test Loss', test_loss.result(), step=epoch)
        tf.summary.scalar('Train Accuracy', train_accuracy.result(), step=epoch)
        tf.summary.scalar('Test Accuracy', test_accuracy.result(), step=epoch)

    message = (
        f'Epoch {epoch + 1}, '
        f'Train Loss: {train_loss.result()}, '
        f'Test Loss: {test_loss.result()}, '
        f'Train Accuracy: {train_accuracy.result()}, '
        f'Test Accuracy: {test_accuracy.result()}'
    )
   
    print(message)

    # Store the loss and accuracy values for plotting
    train_losses.append(train_loss.result().numpy())
    test_losses.append(test_loss.result().numpy())
    train_accuracies.append(train_accuracy.result().numpy())
    test_accuracies.append(test_accuracy.result().numpy())
  
# Plot the training and validation loss

# Plot the training and validation loss
plt.style.use('ggplot')
plt.figure(figsize=(10, 4))

# Loss Plot
plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Train Loss")
plt.plot(test_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

# Accuracy Plot
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label="Train Accuracy")
plt.plot(test_accuracies, label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()
