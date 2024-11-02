import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Define the directory path
data_dir = "C:/Users/Sheekha/Downloads/archive (1)/Dataset_BUSI_with_GT"
print("Data loaded successfully")

# Initialize data and labels
X = []
y = []

# Define label mappings for each folder
label_map = {'normal': 0, 'benign': 1, 'malignant': 2}

# Load images and labels
for folder, label in label_map.items():
    folder_path = os.path.join(data_dir, folder)
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        try:
            # Open image and ensure it's in RGB
            img = Image.open(img_path).convert('RGB')
            img = img.resize((224, 224))  # Resize to match model input
            X.append(np.array(img))
            y.append(label)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")

# Convert data to NumPy arrays and normalize
X = np.array(X).astype('float32') / 255.0
y = np.array(y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(3, activation='softmax')  # 3 classes: normal, benign, malignant
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Callbacks for early stopping and model checkpoint
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True)

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping, checkpoint])

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print('Test Loss:', loss)
print('Test Accuracy:', accuracy)

# Prediction function
def predict_breast_cancer(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict class
    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction[0])

    # Map the prediction to class label
    label_map_inv = {0: 'Normal', 1: 'Benign', 2: 'Malignant'}
    return label_map_inv[class_idx]

# Example usage
input_image = "D:/inputforbreast1.png"
prediction = predict_breast_cancer(input_image)
print("Prediction:", prediction)
