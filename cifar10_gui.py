import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model("cnn_cifar10.h5")
classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((32, 32))  # Resize to 32x32
    img = np.array(img) / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
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
        result_label.config(text=f"Predicted Class: {classes[class_index]}")

# GUI Application
root = tk.Tk()
root.title("CIFAR-10 Classifier")

load_button = Button(root, text="Load Image", command=load_image)
load_button.pack(pady=10)

image_label = Label(root)
image_label.pack()

result_label = Label(root, text="")
result_label.pack(pady=10)

root.mainloop()
