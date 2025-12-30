import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model("mnist_cnn.h5")

def preprocess_image(image_path):
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    img = img.resize((28, 28))  # Resize to 28x28
    img = np.array(img) / 255.0  # Normalize
    img = np.expand_dims(img, axis=(0, -1))  # Add batch and channel dimensions
    return img

def predict_image(image_path):
    img = preprocess_image(image_path)
    prediction = model.predict(img)
    return np.argmax(prediction)

def load_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        img = Image.open(file_path)
        img = img.resize((128, 128))
        img = ImageTk.PhotoImage(img)
        image_label.config(image=img)
        image_label.image = img

        class_index = predict_image(file_path)
        result_label.config(text=f"Predicted Digit: {class_index}")

# GUI Application
root = tk.Tk()
root.title("MNIST Digit Classifier")

load_button = Button(root, text="Load Image", command=load_image)
load_button.pack(pady=10)

image_label = Label(root)
image_label.pack()

result_label = Label(root, text="")
result_label.pack(pady=10)

root.mainloop()
