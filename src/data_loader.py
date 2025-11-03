"""
Moduł do ładowania i przygotowania danych treningowych
"""
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from PIL import Image
import config


def load_images_from_directory(directory, classes):
    """
    Ładuje obrazy z katalogów klas i zwraca dane oraz etykiety
    
    Args:
        directory: Ścieżka do katalogu z podkatalogami klas
        classes: Lista nazw klas
    
    Returns:
        images: Tablica numpy z obrazami
        labels: Tablica numpy z etykietami
    """
    images = []
    labels = []
    
    for class_idx, class_name in enumerate(classes):
        class_dir = os.path.join(directory, class_name)
        if not os.path.exists(class_dir):
            print(f"Ostrzeżenie: Katalog {class_dir} nie istnieje. Pomijam.")
            continue
            
        for filename in os.listdir(class_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                img_path = os.path.join(class_dir, filename)
                try:
                    img = Image.open(img_path)
                    # Konwertuj na RGB jeśli potrzeba
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    # Zmień rozmiar
                    img = img.resize(config.IMAGE_SIZE)
                    img_array = np.array(img) / 255.0  # Normalizacja do [0, 1]
                    images.append(img_array)
                    labels.append(class_idx)
                except Exception as e:
                    print(f"Błąd przy ładowaniu {img_path}: {e}")
                    continue
    
    return np.array(images), np.array(labels)


def get_data_generators(train_dir, validation_split=0.2, batch_size=32):
    """
    Tworzy generatory danych z augmentacją dla treningu i walidacji
    
    Args:
        train_dir: Ścieżka do katalogu treningowego
        validation_split: Proporcja danych do walidacji
        batch_size: Rozmiar batcha
    
    Returns:
        train_generator: Generator danych treningowych
        validation_generator: Generator danych walidacyjnych
    """
    # Data augmentation dla danych treningowych
    train_datagen = ImageDataGenerator(
        rotation_range=config.ROTATION_RANGE,
        width_shift_range=config.WIDTH_SHIFT_RANGE,
        height_shift_range=config.HEIGHT_SHIFT_RANGE,
        brightness_range=config.BRIGHTNESS_RANGE,
        horizontal_flip=config.HORIZONTAL_FLIP,
        zoom_range=config.ZOOM_RANGE,
        validation_split=validation_split,
        rescale=1./255  # Normalizacja
    )
    
    # Generator danych walidacyjnych (bez augmentacji)
    validation_datagen = ImageDataGenerator(
        validation_split=validation_split,
        rescale=1./255
    )
    
    # Generator danych treningowych
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=config.IMAGE_SIZE,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    # Generator danych walidacyjnych
    validation_generator = validation_datagen.flow_from_directory(
        train_dir,
        target_size=config.IMAGE_SIZE,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    return train_generator, validation_generator


def get_test_data_generator(test_dir, batch_size=32):
    """
    Tworzy generator danych testowych (bez augmentacji)
    
    Args:
        test_dir: Ścieżka do katalogu testowego
        batch_size: Rozmiar batcha
    
    Returns:
        test_generator: Generator danych testowych
    """
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=config.IMAGE_SIZE,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    return test_generator


def preprocess_image(image_path):
    """
    Przetwarza pojedynczy obraz do formatu używanego przez model
    
    Args:
        image_path: Ścieżka do obrazu
    
    Returns:
        Preprocessed image jako numpy array
    """
    img = Image.open(image_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(config.IMAGE_SIZE)
    img_array = np.array(img) / 255.0
    # Dodaj wymiar batch
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


if __name__ == "__main__":
    # Test ładowania danych
    print("Testowanie ładowania danych...")
    train_gen, val_gen = get_data_generators(config.TRAIN_DIR)
    print(f"Liczba klas: {train_gen.num_classes}")
    print(f"Liczba próbek treningowych: {train_gen.samples}")
    print(f"Liczba próbek walidacyjnych: {val_gen.samples}")
    print(f"Nazwy klas: {train_gen.class_indices}")

