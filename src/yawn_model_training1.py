import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, InputLayer
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ------------------------ Project Configuration ------------------------
print("Setting up project configuration...")
PROJECT_PATH = "/home/dhruv/Driver_Drowsiness_detection(CNN multithreading)/"
YAWN_DATA_PATH = os.path.join(PROJECT_PATH, "data", "yawn_data")
MODEL_DIR = os.path.join(PROJECT_PATH, "models")
os.makedirs(MODEL_DIR, exist_ok=True)
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_yawn_model.h5")
TFLITE_MODEL_PATH = os.path.join(MODEL_DIR, "yawn_model.tflite")
CONF_MATRIX_IMAGE = os.path.join(MODEL_DIR, "confusion_matrix.png")
TRAIN_HISTORY_IMAGE = os.path.join(MODEL_DIR, "training_history.png")

# Verify data existence
if not os.path.exists(YAWN_DATA_PATH):
    raise FileNotFoundError(f"Yawn data path {YAWN_DATA_PATH} does not exist")

# ------------------------ Data Generators with Augmentation ------------------------
print("Initializing data generators...")
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2  # Split data into training and validation sets
)

# For training and validation, we use the same generator with subset parameter
train_generator = train_datagen.flow_from_directory(
    YAWN_DATA_PATH,
    target_size=(64, 32),   # Height x Width (grayscale images)
    batch_size=32,
    color_mode='grayscale',
    class_mode='binary',
    subset='training'
)
val_generator = train_datagen.flow_from_directory(
    YAWN_DATA_PATH,
    target_size=(64, 32),
    batch_size=32,
    color_mode='grayscale',
    class_mode='binary',
    subset='validation'
)

# For test evaluation, we use a generator without augmentation
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    YAWN_DATA_PATH,
    target_size=(64, 32),
    batch_size=32,
    color_mode='grayscale',
    class_mode='binary',
    shuffle=False,
    subset='validation'  # Using the validation subset for test evaluation here
)

# ------------------------ Enhanced Model Architecture ------------------------
print("Building the model architecture...")
model = Sequential([
    # Input Layer
    InputLayer(input_shape=(64, 32, 1)),
    
    # Block 1
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    
    # Block 2
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    
    # Block 3
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    
    # Additional Block for Improved Feature Extraction
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile the model with a carefully chosen learning rate for better convergence
optimizer = Adam(learning_rate=0.0005)
model.compile(optimizer=optimizer,
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

# ------------------------ Callbacks Setup ------------------------
print("Configuring callbacks...")
callbacks = [
    EarlyStopping(patience=10, restore_best_weights=True, monitor='val_loss', verbose=1),
    ModelCheckpoint(BEST_MODEL_PATH, save_best_only=True, verbose=1)
]

# ------------------------ Model Training ------------------------
print("Training the model... This might take a while!")
history = model.fit(
    train_generator,
    epochs=30,
    validation_data=val_generator,
    callbacks=callbacks,
    verbose=1
)

# ------------------------ Training History Visualization ------------------------
print("Plotting training history...")
plt.figure(figsize=(14, 6))

# Accuracy Plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy', marker='o')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='o')
plt.title("Training vs. Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

# Loss Plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss', marker='o')
plt.plot(history.history['val_loss'], label='Validation Loss', marker='o')
plt.title("Training vs. Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(TRAIN_HISTORY_IMAGE)
plt.show()

# ------------------------ Evaluate on Test Set ------------------------
print("Evaluating on the test set...")
test_loss, test_accuracy = model.evaluate(test_generator, verbose=1)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# ------------------------ Predictions and Classification Report ------------------------
print("Generating predictions on test set...")
Y_pred = model.predict(test_generator, verbose=1)
y_pred = np.where(Y_pred > 0.5, 1, 0).flatten()
class_report = classification_report(test_generator.classes, y_pred)
print("Classification Report:")
print(class_report)

# Calculate overall accuracy using sklearn (optional)
calc_accuracy = accuracy_score(test_generator.classes, y_pred)
print(f"Calculated Test Accuracy: {calc_accuracy * 100:.2f}%")

# ------------------------ Confusion Matrix ------------------------
print("Generating confusion matrix...")
cm = confusion_matrix(test_generator.classes, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=['Predicted No Yawn', 'Predicted Yawn'],
            yticklabels=['Actual No Yawn', 'Actual Yawn'])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.savefig(CONF_MATRIX_IMAGE)
plt.show()

# ------------------------ Save the Model in TensorFlow Lite Format ------------------------
print("Converting the model to TensorFlow Lite format...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open(TFLITE_MODEL_PATH, 'wb') as f:
    f.write(tflite_model)
print(f"TensorFlow Lite model saved to: {TFLITE_MODEL_PATH}")

print("All steps completed successfully!")
