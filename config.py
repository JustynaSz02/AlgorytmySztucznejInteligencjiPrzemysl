"""
Konfiguracja systemu rozpoznawania obiektów
"""
import os

# Ścieżki do danych
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Klasy obiektów do rozpoznawania
# UWAGA: Kolejność musi być alfabetyczna, aby zgadzała się z ImageDataGenerator
CLASSES = [
    "blue_circle",
    "blue_square",
    "green_circle",
    "green_square",
    "red_circle",
    "red_square"
]

NUM_CLASSES = len(CLASSES)

# Parametry obrazów
IMAGE_SIZE = (224, 224)  # Rozmiar obrazu wejściowego (szerokość, wysokość)
IMAGE_CHANNELS = 3  # RGB

# Parametry treningu
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2  # 20% danych do walidacji

# Parametry data augmentation
ROTATION_RANGE = 20  # Stopnie rotacji
WIDTH_SHIFT_RANGE = 0.1  # Przesunięcie w poziomie (10%)
HEIGHT_SHIFT_RANGE = 0.1  # Przesunięcie w pionie (10%)
BRIGHTNESS_RANGE = [0.8, 1.2]  # Zakres zmiany jasności
HORIZONTAL_FLIP = True
ZOOM_RANGE = 0.1  # Zakres powiększenia

# Parametry modelu
BASE_MODEL = "MobileNetV2"  # MobileNetV2 lub EfficientNetB0
DROPOUT_RATE = 0.5

# Parametry callbacków
EARLY_STOPPING_PATIENCE = 10  # Zatrzymaj trening jeśli brak poprawy przez 10 epok
REDUCE_LR_PATIENCE = 5  # Zmniejsz learning rate jeśli brak poprawy przez 5 epok
REDUCE_LR_FACTOR = 0.5  # Współczynnik redukcji learning rate
MIN_LEARNING_RATE = 1e-7

# Nazwa pliku modelu
MODEL_NAME = "object_classifier_model.h5"
MODEL_PATH = os.path.join(MODELS_DIR, MODEL_NAME)

# Tworzenie katalogów jeśli nie istnieją
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

