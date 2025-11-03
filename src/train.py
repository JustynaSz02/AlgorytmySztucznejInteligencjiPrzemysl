"""
Skrypt treningowy do uczenia modelu rozpoznawania obiektów
"""
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    CSVLogger
)
# Dodaj główny katalog do ścieżki Python
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

import config
from src.data_loader import get_data_generators
from src.model import create_model, compile_model, save_model, get_model_summary


def create_callbacks():
    """
    Tworzy callbacki do użycia podczas treningu
    
    Returns:
        Lista callbacków
    """
    callbacks = [
        # Early stopping - zatrzymaj jeśli brak poprawy
        EarlyStopping(
            monitor='val_loss',
            patience=config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Zapisz najlepszy model
        ModelCheckpoint(
            filepath=config.MODEL_PATH,
            monitor='val_loss',
            save_best_only=True,
            verbose=1,
            mode='min'
        ),
        
        # Zmniejsz learning rate jeśli brak poprawy
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=config.REDUCE_LR_FACTOR,
            patience=config.REDUCE_LR_PATIENCE,
            min_lr=config.MIN_LEARNING_RATE,
            verbose=1
        ),
        
        # Zapisz historię treningu do CSV
        CSVLogger(
            filename=os.path.join(config.MODELS_DIR, 'training_log.csv'),
            append=False
        )
    ]
    
    return callbacks


def plot_training_history(history, save_path=None):
    """
    Wizualizuje historię treningu (accuracy i loss)
    
    Args:
        history: Historia treningu z model.fit()
        save_path: Ścieżka do zapisania wykresów (opcjonalnie)
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Wykres accuracy
    axes[0].plot(history.history['accuracy'], label='Trening', marker='o')
    axes[0].plot(history.history['val_accuracy'], label='Walidacja', marker='s')
    axes[0].set_title('Dokładność modelu')
    axes[0].set_xlabel('Epoka')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    
    # Wykres loss
    axes[1].plot(history.history['loss'], label='Trening', marker='o')
    axes[1].plot(history.history['val_loss'], label='Walidacja', marker='s')
    axes[1].set_title('Funkcja kosztu')
    axes[1].set_xlabel('Epoka')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Wykresy zapisane w: {save_path}")
    
    plt.show()


def train():
    """
    Główna funkcja treningowa
    """
    print("=" * 60)
    print("ROZPOCZĘCIE TRENINGU MODELU")
    print("=" * 60)
    
    # Sprawdź czy istnieją dane treningowe
    if not os.path.exists(config.TRAIN_DIR):
        raise FileNotFoundError(
            f"Katalog z danymi treningowymi nie istnieje: {config.TRAIN_DIR}\n"
            f"Upewnij się, że dodałeś obrazy do odpowiednich folderów klas."
        )
    
    # Przygotuj generatory danych
    print("\nPrzygotowywanie danych...")
    train_generator, validation_generator = get_data_generators(
        config.TRAIN_DIR,
        validation_split=config.VALIDATION_SPLIT,
        batch_size=config.BATCH_SIZE
    )
    
    print(f"Liczba próbek treningowych: {train_generator.samples}")
    print(f"Liczba próbek walidacyjnych: {validation_generator.samples}")
    print(f"Liczba klas: {train_generator.num_classes}")
    print(f"Mapowanie klas: {train_generator.class_indices}")
    
    # Utwórz model
    print("\nTworzenie modelu...")
    model = create_model(pretrained=True)
    model = compile_model(model)
    
    print("\nArchitektura modelu:")
    get_model_summary(model)
    
    # Przygotuj callbacki
    callbacks = create_callbacks()
    
    # Rozpocznij trening
    print("\n" + "=" * 60)
    print("ROZPOCZĘCIE TRENINGU")
    print("=" * 60)
    print(f"Parametry treningu:")
    print(f"  - Batch size: {config.BATCH_SIZE}")
    print(f"  - Maksymalna liczba epok: {config.EPOCHS}")
    print(f"  - Learning rate: {config.LEARNING_RATE}")
    print(f"  - Validation split: {config.VALIDATION_SPLIT}")
    print()
    
    history = model.fit(
        train_generator,
        epochs=config.EPOCHS,
        validation_data=validation_generator,
        callbacks=callbacks,
        verbose=1
    )
    
    # Zapisz końcowy model
    print("\nZapisywanie modelu...")
    save_model(model)
    
    # Wizualizuj wyniki
    print("\nTworzenie wizualizacji wyników...")
    plot_save_path = os.path.join(config.MODELS_DIR, 'training_history.png')
    plot_training_history(history, save_path=plot_save_path)
    
    # Podsumowanie
    print("\n" + "=" * 60)
    print("TRENING ZAKOŃCZONY")
    print("=" * 60)
    print(f"Ostatnia dokładność treningowa: {history.history['accuracy'][-1]:.4f}")
    print(f"Ostatnia dokładność walidacyjna: {history.history['val_accuracy'][-1]:.4f}")
    print(f"Ostatni loss treningowy: {history.history['loss'][-1]:.4f}")
    print(f"Ostatni loss walidacyjny: {history.history['val_loss'][-1]:.4f}")
    print(f"\nModel zapisany w: {config.MODEL_PATH}")


if __name__ == "__main__":
    train()

