import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tkinter import *
from PIL import Image, ImageDraw, ImageTk
import os
import random
from tensorflow.keras.utils import plot_model
from scipy.ndimage import center_of_mass

# Function to create and train the model
def create_and_train_model():
    # Load and prepare the MNIST dataset
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Invert the colors
    x_train = 1 - x_train
    x_test = 1 - x_test

    # Reshape data for data augmentation (add channel dimension)
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

    # Data Augmentation
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.2),
        tf.keras.layers.RandomContrast(0.2),
    ])

    # Build the model
    model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
        data_augmentation,
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(10)
    ])

    # Compile the model with a learning rate scheduler
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.001,
        decay_steps=10000,
        decay_rate=0.9)
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # Use early stopping to prevent overfitting
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

    # Train the model for more epochs
    model.fit(x_train, y_train, epochs=20, validation_data=(x_test, y_test), callbacks=[early_stopping])

    # Save the model in the current working directory
    model.save('mnist_model_improved.keras')
    print("Model created and saved as mnist_model_improved.keras")

    # Save test data for later use
    np.save('x_train.npy', x_train)
    np.save('y_train.npy', y_train)
    np.save('x_test.npy', x_test)
    np.save('y_test.npy', y_test)
    print("Training and test data saved.")

    # Visualize the model
    plot_model(model, to_file='model_architecture_improved.png', show_shapes=True, show_layer_names=True)
    print("Model architecture saved as model_architecture_improved.png")

def display_sample_data(x_train, y_train):
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(x_train[i].reshape(28, 28), cmap=plt.cm.binary)
        plt.xlabel(y_train[i])
    plt.show()

def display_random_digits_with_predictions(model):
    if not os.path.exists('x_test.npy') or not os.path.exists('y_test.npy'):
        print("Test data not found. Please create a model first.")
        return

    x_test = np.load('x_test.npy')
    y_test = np.load('y_test.npy')
    indices = random.sample(range(len(x_test)), 5)
    for idx in indices:
        img = x_test[idx]
        plt.imshow(img.reshape(28, 28), cmap='gray')
        plt.show()
        img = img.reshape(1, 28, 28, 1)
        prediction = model.predict(img)
        predicted_digit = np.argmax(prediction)
        actual_digit = y_test[idx]
        print(f"Predicted Digit: {predicted_digit}, Actual Digit: {actual_digit}")

def center_and_pad_image(img_array):
    # Calculate the center of mass of the digit
    cy, cx = center_of_mass(1 - img_array)
    shift_x = int(14 - cx)
    shift_y = int(14 - cy)
    img_array = np.roll(img_array, shift_y, axis=0)
    img_array = np.roll(img_array, shift_x, axis=1)
    return img_array

# Create the GUI for drawing a digit

class ImageDrawer:
    def __init__(self, root, model, x_train, y_train):
        self.root = root
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.canvas_size = 280
        self.grid_size = 28
        self.pixel_size = self.canvas_size // self.grid_size
        self.canvas = Canvas(root, width=self.canvas_size, height=self.canvas_size, bg='white')
        self.canvas.pack()
        self.image = Image.new("L", (self.grid_size, self.grid_size), "white")
        self.draw = ImageDraw.Draw(self.image)
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.paint)
        button_predict = Button(text="Predict", command=self.predict)
        button_predict.pack()
        button_clear = Button(text="Clear", command=self.clear)
        button_clear.pack()
        button_show_samples = Button(text="Show Samples", command=self.show_samples)
        button_show_samples.pack()
        button_load_model = Button(text="Load Model", command=self.load_model)
        button_load_model.pack()

    def paint(self, event):
        x = event.x / self.pixel_size
        y = event.y / self.pixel_size
        radius = 2.0  # Floating point radius for a more precise brush size
        sigma = 0.5  # Standard deviation for Gaussian blur
        
        for i in range(int(x - radius * 2), int(x + radius * 2) + 1):
            for j in range(int(y - radius * 2), int(y + radius * 2) + 1):
                dist = ((i - x)**2 + (j - y)**2)**0.5
                if dist <= radius:
                    # Calculate opacity based on Gaussian function
                    opacity = int(255 * np.exp(-(dist**2) / (2 * sigma**2)))
                    # Get the current pixel value
                    if 0 <= i < self.grid_size and 0 <= j < self.grid_size:
                        current_pixel = self.image.getpixel((i, j))
                        # Mix the current pixel value with the new opacity
                        new_pixel_value = max(0, current_pixel - opacity)
                        self.draw.point((i, j), fill=new_pixel_value)

        self.update_canvas()

    def update_canvas(self):
        self.tk_image = ImageTk.PhotoImage(self.image.resize((self.canvas_size, self.canvas_size), Image.NEAREST))
        self.canvas.create_image(0, 0, anchor=NW, image=self.tk_image)

    def predict(self):
        # Resize image to 28x28 pixels
        img = self.image.resize((28, 28), Image.LANCZOS)
        img_array = np.array(img)

        # Normalize the image
        img_array = img_array / 255.0  # Normalize to [0, 1]

        # Center and pad the image
        img_array = center_and_pad_image(img_array)

        # Display the processed image
        display_drawing = False
        if(display_drawing): 
            plt.imshow(img_array, cmap='gray')
            plt.title("Processed Image")
            plt.show()

        img_array = img_array.reshape(1, 28, 28, 1)  # Reshape for model

        # Predict the digit
        prediction = self.model.predict(img_array)
        predicted_digit = np.argmax(prediction)
        print("Predicted Digit:", predicted_digit)

    def clear(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (self.grid_size, self.grid_size), "white")
        self.draw = ImageDraw.Draw(self.image)

    def load_model(self):
        self.model = tf.keras.models.load_model('mnist_model_improved.keras')
        print("Model loaded successfully")

    def show_samples(self):
        display_sample_data(self.x_train, self.y_train)

def evaluate_model_on_test_data(model_path='mnist_model_improved.keras', test_data_path='x_test.npy', test_labels_path='y_test.npy'):
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Please create and save a model first.")
        return

    if not os.path.exists(test_data_path) or not os.path.exists(test_labels_path):
        print(f"Test data not found at {test_data_path} or {test_labels_path}. Please save the test data first.")
        return

    # Load the model
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully")

    # Load test data
    x_test = np.load(test_data_path)
    y_test = np.load(test_labels_path)

    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

    return test_accuracy

def main():
    choice = input("Enter '1' to create a new model, '2' to use the drawing program, '3' to see random digit predictions, or '4' to evaluate the model on test data: ")
    if choice == '1':
        create_and_train_model()
    elif choice == '2':
        if not os.path.exists('mnist_model_improved.keras'):
            print("Model not found. Please create a model first.")
            return
        if not os.path.exists('x_train.npy') or not os.path.exists('y_train.npy'):
            print("Training data not found. Please create a model first.")
            return
        model = tf.keras.models.load_model('mnist_model_improved.keras')
        x_train = np.load('x_train.npy')
        y_train = np.load('y_train.npy')
        root = Tk()
        root.title("Digit Recognizer")
        app = ImageDrawer(root, model, x_train, y_train)
        root.mainloop()
    elif choice == '3':
        if not os.path.exists('mnist_model_improved.keras'):
            print("Model not found. Please create a model first.")
            return
        model = tf.keras.models.load_model('mnist_model_improved.keras')
        display_random_digits_with_predictions(model)
    elif choice == '4':
        evaluate_model_on_test_data()
    else:
        print("Invalid choice. Please enter '1', '2', '3', or '4'.")

if __name__ == "__main__":
    main()

