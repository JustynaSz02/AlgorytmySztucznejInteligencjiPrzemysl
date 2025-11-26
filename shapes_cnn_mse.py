
import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam


BASE_DATA_DIR = "data"                
TRAIN_DIR = os.path.join(BASE_DATA_DIR, "train")
VAL_DIR   = os.path.join(BASE_DATA_DIR, "val")
TEST_DIR  = os.path.join(BASE_DATA_DIR, "test")

IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-3


train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
)

val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

num_classes = train_generator.num_classes
print("Klasy:", train_generator.class_indices)



model = Sequential([
    Conv2D(32, (3, 3), activation='relu',
           input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    MaxPooling2D((2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy', 'mse']   # MSE dodane jako metryka
)

model.summary()



history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    verbose=1
)



test_loss, test_acc, test_mse = model.evaluate(test_generator, verbose=1)
print(f"\nWyniki na TEŚCIE:")
print(f"  loss = {test_loss:.4f}")
print(f"  accuracy = {test_acc:.4f}")
print(f"  MSE = {test_mse:.4f}")



epochs_range = range(1, len(history.history['loss']) + 1)

plt.figure(figsize=(15, 4))

# Accuracy
plt.subplot(1, 3, 1)
plt.plot(epochs_range, history.history['accuracy'], label='train acc')
plt.plot(epochs_range, history.history['val_accuracy'], label='val acc')
plt.title('Dokładność (accuracy)')
plt.xlabel('Epoka')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Loss
plt.subplot(1, 3, 2)
plt.plot(epochs_range, history.history['loss'], label='train loss')
plt.plot(epochs_range, history.history['val_loss'], label='val loss')
plt.title('Funkcja kosztu (loss)')
plt.xlabel('Epoka')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# MSE
plt.subplot(1, 3, 3)
plt.plot(epochs_range, history.history['mse'], label='train MSE')
plt.plot(epochs_range, history.history['val_mse'], label='val MSE')
plt.title('Błąd średniokwadratowy (MSE)')
plt.xlabel('Epoka')
plt.ylabel('MSE')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()



os.makedirs("models", exist_ok=True)
model.save("models/shapes_cnn_mse.h5")
print("Model zapisany w: models/shapes_cnn_mse.h5")
