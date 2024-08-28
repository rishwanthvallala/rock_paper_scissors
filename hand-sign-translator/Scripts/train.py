import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Define constants
image_size = 300
folders = ["Data/Rock", "Data/Paper", "Data/Scissor"]

# Initialize data and labels lists
data = []
labels = []

# Load and preprocess the data
for idx, folder in enumerate(folders):
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (image_size, image_size))  # Resize to 300x300
        data.append(img)
        labels.append(idx)  # Assign a label based on folder

# Convert to numpy arrays
data = np.array(data, dtype="float32") / 255.0  # Normalize the data
labels = np.array(labels)

# One-hot encode the labels
labels = to_categorical(labels, num_classes=3)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Define the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(image_size, image_size, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(3, activation='softmax')  # Output layer with 3 classes
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=32)

# Save the trained model as a .h5 file
model.save('gesture_recognition_model.h5')

# Load the model (if needed later)
model = load_model('gesture_recognition_model.h5')

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test accuracy: {accuracy * 100:.2f}%')

# Example: Make a prediction on a test image
prediction = model.predict(np.expand_dims(X_test[0], axis=0))
predicted_class = np.argmax(prediction, axis=1)
print(f'Predicted class: {predicted_class[0]}')
