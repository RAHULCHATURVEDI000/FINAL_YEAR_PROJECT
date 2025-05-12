import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, InputLayer
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ------------------------ Project Configuration ------------------------
PROJECT_PATH = "/home/dhruv/Driver_Drowsiness_detection(CNN multithreading)/"
EYE_DATA_PATH = os.path.join(PROJECT_PATH, "data", "eye_data")
MODEL_SAVE_PATH = os.path.join(PROJECT_PATH, 'models', 'eye_model.h5')
CONFUSION_MATRIX_IMAGE = os.path.join(PROJECT_PATH, 'models', 'confusion_matrix.png')

# ------------------------ Data Generators with Augmentation ------------------------
print("Setting up data generators with augmentation...")
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Validation and Test: only rescaling
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# ------------------------ Flow Data from Directories ------------------------
print("Flowing data from directories...")
train_generator = train_datagen.flow_from_directory(
    os.path.join(EYE_DATA_PATH, 'train'),
    target_size=(36, 26),
    batch_size=32,
    color_mode='grayscale',
    class_mode='binary'
)

val_generator = val_datagen.flow_from_directory(
    os.path.join(EYE_DATA_PATH, 'validation'),
    target_size=(36, 26),
    batch_size=32,
    color_mode='grayscale',
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    os.path.join(EYE_DATA_PATH, 'test'),
    target_size=(36, 26),
    batch_size=32,
    color_mode='grayscale',
    class_mode='binary',
    shuffle=False
)

# ------------------------ Enhanced Model Architecture ------------------------
print("Building and compiling the model...")

model = Sequential([
    # Input layer
    InputLayer(input_shape=(36, 26, 1)),
    
    # Block 1
    Conv2D(32, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    
    # Block 2
    Conv2D(64, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    
    # Block 3
    Conv2D(128, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    
    # Extra block for improved feature extraction
    Conv2D(128, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile model with a slightly lower learning rate for stability
optimizer = Adam(learning_rate=0.0005)
model.compile(optimizer=optimizer,
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

# ------------------------ Callbacks ------------------------
print("Setting up callbacks...")
callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True, verbose=1),
    ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, verbose=1)
]

# ------------------------ Training ------------------------
print("Starting model training (this may take a while)...")
history = model.fit(
    train_generator,
    epochs=100,
    validation_data=val_generator,
    callbacks=callbacks,
    verbose=1
)

# ------------------------ Plot Training History ------------------------
print("Plotting training history...")
plt.figure(figsize=(12,5))

# Accuracy Plot
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Accuracy', marker='o')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='o')
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

# Loss Plot
plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss', marker='o')
plt.plot(history.history['val_loss'], label='Validation Loss', marker='o')
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(PROJECT_PATH, 'models', 'training_history.png'))
plt.show()

# ------------------------ Evaluate on Test Set ------------------------
print("Evaluating model on test set...")
test_loss, test_accuracy = model.evaluate(test_generator, verbose=1)
print(f"Test Accuracy: {test_accuracy*100:.2f}%")

# ------------------------ Save the Model ------------------------
print("Saving the model...")
model.save(MODEL_SAVE_PATH)

# ------------------------ Predictions and Classification Report ------------------------
print("Generating predictions on the test set...")
Y_pred = model.predict(test_generator, verbose=1)
y_pred = np.where(Y_pred > 0.5, 1, 0).flatten()

print("Classification Report:")
report = classification_report(test_generator.classes, y_pred)
print(report)

# ------------------------ Confusion Matrix ------------------------
print("Generating and saving confusion matrix...")
cm = confusion_matrix(test_generator.classes, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
            xticklabels=['Predicted Closed', 'Predicted Open'],
            yticklabels=['Actual Closed', 'Actual Open'])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.savefig(CONFUSION_MATRIX_IMAGE)
plt.show()

# ------------------------ Additional Evaluation Metrics ------------------------
accuracy = accuracy_score(test_generator.classes, y_pred)
print(f"Calculated Test Accuracy: {accuracy*100:.2f}%")

print("All steps completed successfully!")

