import tensorflow as tf
import tensorflow_hub as hub
import os

# Path to save the modified model
model_save_path = './yamnet_model/13_classes_model.h5'

class YamnetModel(tf.keras.layers.Layer):
    def __init__(self, num_classes=13):
        super(YamnetModel, self).__init__()
        # Load the pre-trained YAMNet model from TensorFlow Hub
        self.yamnet_model = hub.KerasLayer('https://tfhub.dev/google/yamnet/1', trainable=False)
        self.dense_layer = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, inputs):
        # Process inputs through YAMNet and then the new classification layer
        embeddings = self.yamnet_model(inputs)
        outputs = self.dense_layer(embeddings)
        return outputs

    def build(self, input_shape):
        # This method can be implemented if required, here it ensures the layer is properly built
        super(YamnetModel, self).build(input_shape)

def modify_yamnet(num_classes=13):
    """
    Modify YAMNet by replacing the final classification layer with a layer that outputs `num_classes`.
    If the model is already saved, load it from disk instead of downloading from TensorFlow Hub.
    """
    if os.path.exists(model_save_path):
        # Load the previously saved model if it exists
        print(f"Loading the saved model from {model_save_path}...")
        model = tf.keras.models.load_model(model_save_path)
    else:
        # Create a new model using the custom layer
        print("Downloading YAMNet model from TensorFlow Hub and modifying it...")
        inputs = tf.keras.Input(shape=(None,), dtype=tf.float32)  # Raw waveform input
        outputs = YamnetModel(num_classes=num_classes)(inputs)  # Use the custom YAMNet layer
        model = tf.keras.Model(inputs, outputs)

        # Save the modified model for future use
        model.save(model_save_path)
        print(f"Model saved to {model_save_path}")

    return model

def freeze_yamnet_layers(model):
    """
    Freeze all layers except the last classification layer.
    """
    for layer in model.layers[:-1]:
        layer.trainable = False

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Prepare the model
model = modify_yamnet(num_classes=13)
freeze_yamnet_layers(model)

# Few-shot training on a small dataset
# X_train: raw waveform audio samples
# y_train: corresponding chord labels (13 classes)
# model.fit(X_train, y_train, epochs=10, batch_size=8)
