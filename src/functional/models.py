import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, loss, optimizer):
        super().__init__()
        self.loss = loss
        self.optimizer = optimizer
        self.conv1 = tf.keras.layers.Conv2D(32, kernel_size=3, activation='relu')
        #self.norm = tf.keras.layers.BatchNormalization()
        self.drop = tf.keras.layers.Dropout(0.5)
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(10)

    def call(self, x): # in pytorch this would be the forward method
        x = self.conv1(x)
        x = self.drop(x)
        x = self.flatten(x)
        x = self.dense1(x)
        return self.dense2(x)

    def train_step(self, images, labels):
        with tf.GradientTape() as tape: #keep grads = True
            predictions = self(images, training=True) # this calls the method call
            loss = self.loss(labels, predictions)
        gradients = tape.gradient(loss, self.trainable_variables) #compute the gradients of the loss in regard to the trainable variables
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables)) #updates our weights zip creates a tuple, must be of the same size!
        return predictions, loss, gradients
        
    def test_step(self, images, labels):
        predictions = self(images, training=False)
        loss = self.loss(labels, predictions)
        return predictions, loss