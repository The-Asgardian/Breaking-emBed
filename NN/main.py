# gesture_recognition_fixed.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from keras._tf_keras.keras import Sequential
from keras._tf_keras.keras.layers import Conv1D, MaxPooling1D, Dense, Flatten, Dropout
from keras._tf_keras.keras.utils import to_categorical
from keras._tf_keras.keras.callbacks import EarlyStopping
import tensorflow as tf

# Configuration
np.random.seed(42)
tf.random.set_seed(42)

# Parameters
NUM_GESTURES = 5              # Number of gesture classes
SAMPLES_PER_GESTURE = 150     # 3 seconds at 50Hz
NUM_CHANNELS = 6              # 3 accel + 3 gyro
INSTANCES_PER_CLASS = 20      # Number of examples per gesture class

# Generate dummy data for testing
def generate_dummy_data():
    X = []
    y = []
    
    for gesture_id in range(NUM_GESTURES):
        for _ in range(INSTANCES_PER_CLASS):
            # Base time array
            time = np.linspace(0, 2*np.pi, SAMPLES_PER_GESTURE)
            
            # Create unique pattern for each gesture
            pattern = np.zeros((SAMPLES_PER_GESTURE, NUM_CHANNELS))
            for i in range(NUM_CHANNELS):
                # Gesture-specific pattern generation
                freq = (gesture_id + 1) * (i + 1) * 0.5
                phase = np.random.uniform(0, 2*np.pi)  # Random phase per instance
                amplitude = np.random.uniform(0.8, 1.2)  # Random amplitude
                pattern[:, i] = amplitude * np.sin(freq * time + phase)
            
            # Add noise and normalize
            noise = np.random.normal(0, 0.2, (SAMPLES_PER_GESTURE, NUM_CHANNELS))
            data = pattern + noise
            
            X.append(data)
            y.append(gesture_id)
    
    return np.array(X), np.array(y)

# Generate and split data
X, y = generate_dummy_data()
X_train, X_test, y_train, y_test = train_test_split(
X, y, 
test_size=0.2, 
stratify=y,
random_state=42
)

# Normalize data (I might want to split this up into training and testing separately)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train.reshape(-1, NUM_CHANNELS)).reshape(X_train.shape)
X_test = scaler.transform(X_test.reshape(-1, NUM_CHANNELS)).reshape(X_test.shape)

# One-hot encode labels
y_train = to_categorical(y_train, NUM_GESTURES)
y_test = to_categorical(y_test, NUM_GESTURES)

# Build 1D CNN model  (Maybe add more layers if this is not enough?????)
model = Sequential([
Conv1D(64, 3, activation='relu', input_shape=(SAMPLES_PER_GESTURE, NUM_CHANNELS)),
MaxPooling1D(2),
Conv1D(128, 3, activation='relu'),
MaxPooling1D(2),
Flatten(),
Dense(128, activation='relu'),
Dropout(0.5),                               # Maybe change to L2 regularization
Dense(NUM_GESTURES, activation='softmax')   # Maybe change to sigmoid
])

# Maybe add batch normalization????

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

# Train model
history = model.fit(X_train, y_train,epochs=50,batch_size=32,validation_split=0.2,callbacks=[EarlyStopping(patience=5, restore_best_weights=True)])

# Evaluate model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"\nTest Accuracy: {test_acc*100:.2f}%")


# Plot training history 
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training History')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()

# Confusion matrix
plt.subplot(1, 2, 2)
y_pred = model.predict(X_test).argmax(axis=1)
y_true = y_test.argmax(axis=1)
conf_matrix = tf.math.confusion_matrix(y_true, y_pred)
plt.imshow(conf_matrix, cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.colorbar()
plt.tight_layout()
plt.show()

# Save model and scaler
model.save('gesture_model.h5')
np.save('scaler_params.npy', {'mean': scaler.mean_, 'scale': scaler.scale_})

print("Model and scaler saved successfully!")
