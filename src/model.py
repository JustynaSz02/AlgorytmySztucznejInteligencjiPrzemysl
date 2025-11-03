"""
Architektura modelu do rozpoznawania obiektów
Używa transfer learning z MobileNetV2
"""
import os
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import config


def create_model(pretrained=True):
    """
    Tworzy model oparty na MobileNetV2 z dostosowanymi warstwami wyjściowymi
    
    Args:
        pretrained: Czy użyć wstępnie wytrenowanego modelu ImageNet
    
    Returns:
        Skompilowany model Keras
    """
    # Bazowy model MobileNetV2 (bez warstw klasyfikacji)
    if pretrained:
        base_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=(*config.IMAGE_SIZE, config.IMAGE_CHANNELS)
        )
    else:
        base_model = MobileNetV2(
            weights=None,
            include_top=False,
            input_shape=(*config.IMAGE_SIZE, config.IMAGE_CHANNELS)
        )
    
    # Zamrożenie warstw bazowego modelu (opcjonalnie można odblokować ostatnie warstwy)
    base_model.trainable = True
    
    # Odblokuj ostatnie kilka warstw do fine-tuningu
    fine_tune_at = len(base_model.layers) - 20
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
    
    # Budowanie modelu
    inputs = base_model.input
    x = base_model.output
    
    # Global Average Pooling zamiast Flatten
    x = GlobalAveragePooling2D()(x)
    
    # Warstwa Dropout dla regularyzacji
    x = Dropout(config.DROPOUT_RATE)(x)
    
    # Warstwa wyjściowa
    outputs = Dense(config.NUM_CLASSES, activation='softmax', name='predictions')(x)
    
    # Utworzenie modelu
    model = Model(inputs=inputs, outputs=outputs)
    
    return model


def compile_model(model, learning_rate=None):
    """
    Kompiluje model z optymalizatorem i metrykami
    
    Args:
        model: Model Keras do skompilowania
        learning_rate: Learning rate dla optymalizatora (domyślnie z config)
    
    Returns:
        Skompilowany model
    """
    if learning_rate is None:
        learning_rate = config.LEARNING_RATE
    
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def load_model(model_path=None):
    """
    Ładuje wytrenowany model z dysku
    
    Args:
        model_path: Ścieżka do pliku modelu (domyślnie z config)
    
    Returns:
        Wczytany model Keras
    """
    from tensorflow.keras.models import load_model as keras_load_model
    
    if model_path is None:
        model_path = config.MODEL_PATH
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model nie został znaleziony w ścieżce: {model_path}")
    
    model = keras_load_model(model_path)
    return model


def save_model(model, model_path=None):
    """
    Zapisuje model na dysk
    
    Args:
        model: Model Keras do zapisania
        model_path: Ścieżka do zapisania modelu (domyślnie z config)
    """
    if model_path is None:
        model_path = config.MODEL_PATH
    
    # Upewnij się, że katalog istnieje
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    model.save(model_path)
    print(f"Model zapisany w: {model_path}")


def get_model_summary(model):
    """
    Wyświetla podsumowanie architektury modelu
    
    Args:
        model: Model Keras
    """
    model.summary()
    
    # Wyświetl informacje o zamrożonych/odblokowanych warstwach
    trainable_count = sum([1 for layer in model.layers if layer.trainable])
    total_count = len(model.layers)
    print(f"\nWarstwy trenowalne: {trainable_count}/{total_count}")


if __name__ == "__main__":
    # Test tworzenia modelu
    print("Tworzenie modelu...")
    model = create_model(pretrained=True)
    model = compile_model(model)
    print("\nPodsumowanie modelu:")
    get_model_summary(model)

