import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

# Load and preprocess MNIST data
(Xtrain, ytrain), (Xtest, ytest) = datasets.mnist.load_data()
Xtrain = Xtrain / 255.0  # Normalize
Xtest = Xtest / 255.0  # Normalize
Xtrain = Xtrain.reshape(-1, 28, 28, 1)  # Add channel dimension
Xtest = Xtest.reshape(-1, 28, 28, 1)

# Build the CNN model
cnn = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile the model
cnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
cnn.fit(Xtrain, ytrain, epochs=10, validation_data=(Xtest, ytest))

# Save the trained model
cnn.save("mnist_cnn.h5")

# Evaluate the model
test_loss, test_accuracy = cnn.evaluate(Xtest, ytest)
print(f"Test Accuracy: {test_accuracy:.2f}")
