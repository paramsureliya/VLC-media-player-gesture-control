import os
import cv2
import numpy as np
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split


# Function to load and preprocess images
def load_and_preprocess_data(data_path, class_labels, img_size=(300, 300)):
    images = []
    labels = []

    for class_label, class_name in enumerate(class_labels):
        class_path = os.path.join(data_path, class_name)
        for filename in os.listdir(class_path):
            img_path = os.path.join(class_path, filename)

            img = cv2.imread(img_path)
            img = cv2.resize(img, img_size)
            img = img.astype('float') / 255.0

            images.append(img)
            labels.append(class_label)

    images = np.array(images)
    labels = np.array(labels)

    images, labels = shuffle(images, labels, random_state=42)

    return images, labels


# Define classes
class_labels = ['play_pause','volume_up', 'volume_down', 'forward', 'backward']

# Load and preprocess the data
data_path = r'C:\Users\PARAM M. SURELIYA\PycharmProjects\sign language\Data'
X, y = load_and_preprocess_data(data_path, class_labels)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert labels to one-hot encoding
num_classes = len(class_labels)
y_train_one_hot = tf.keras.utils.to_categorical(y_train, num_classes)
y_test_one_hot = tf.keras.utils.to_categorical(y_test, num_classes)

# Create a simple CNN model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(300, 300, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
batch_size = 32
epochs = 8
model.fit(X_train, y_train_one_hot, validation_data=(X_test, y_test_one_hot), epochs=epochs, batch_size=batch_size)

# Save the model
model.save('hand_gesture_model.keras')

