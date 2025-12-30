import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

# Load and preprocess CIFAR-10 data
(Xtrain, ytrain), (Xtest, ytest) = datasets.cifar10.load_data()
Xtrain, Xtest = Xtrain / 255.0, Xtest / 255.0  # Normalize
ytrain, ytest = ytrain.reshape(-1,), ytest.reshape(-1,)  # Flatten labels

classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

# Build the CNN model
cnn = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile the model
cnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
cnn.fit(Xtrain, ytrain, epochs=15, validation_data=(Xtest, ytest))

# Save the trained model
cnn.save("cnn_cifar10.h5")

# Evaluate the model
test_loss, test_accuracy = cnn.evaluate(Xtest, ytest)
print(f"Test Accuracy: {test_accuracy:.2f}")
