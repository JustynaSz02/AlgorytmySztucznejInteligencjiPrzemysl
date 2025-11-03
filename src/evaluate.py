"""
Skrypt do ewaluacji wytrenowanego modelu
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)
# Dodaj główny katalog do ścieżki Python
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

import config
from src.model import load_model
from src.data_loader import get_test_data_generator


def evaluate_model(model, test_generator):
    """
    Ewaluuje model na danych testowych
    
    Args:
        model: Wytrenowany model Keras
        test_generator: Generator danych testowych
    
    Returns:
        Słownik z wynikami ewaluacji
    """
    print("Wykonywanie predykcji na danych testowych...")
    
    # Pobierz prawdziwe etykiety i predykcje
    y_true = []
    y_pred = []
    
    # Resetuj generator
    test_generator.reset()
    
    # Pobierz wszystkie predykcje
    num_batches = len(test_generator)
    for i in range(num_batches):
        batch_x, batch_y = test_generator[i]
        batch_pred = model.predict(batch_x, verbose=0)
        
        # Konwertuj one-hot encoding na indeksy klas
        y_true.extend(np.argmax(batch_y, axis=1))
        y_pred.extend(np.argmax(batch_pred, axis=1))
    
    # Konwertuj na numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Oblicz metryki
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    # Średnie ważone (uwzględniając liczebność klas)
    precision_weighted = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )[0]
    recall_weighted = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )[1]
    f1_weighted = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )[2]
    
    # Macierz pomyłek
    cm = confusion_matrix(y_true, y_pred)
    
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted,
        'confusion_matrix': cm,
        'y_true': y_true,
        'y_pred': y_pred,
        'support': support
    }
    
    return results


def plot_confusion_matrix(cm, class_names, save_path=None):
    """
    Wizualizuje macierz pomyłek
    
    Args:
        cm: Macierz pomyłek
        class_names: Nazwy klas
        save_path: Ścieżka do zapisania wykresu (opcjonalnie)
    """
    plt.figure(figsize=(10, 8))
    
    # Normalizuj macierz (pokazuj procenty)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Procent'}
    )
    
    plt.title('Macierz pomyłek (znormalizowana)', fontsize=16, pad=20)
    plt.ylabel('Prawdziwa klasa', fontsize=12)
    plt.xlabel('Przewidywana klasa', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Macierz pomyłek zapisana w: {save_path}")
    
    plt.show()


def plot_class_metrics(precision, recall, f1, class_names, save_path=None):
    """
    Wizualizuje metryki per klasa
    
    Args:
        precision: Tablica precision per klasa
        recall: Tablica recall per klasa
        f1: Tablica F1-score per klasa
        class_names: Nazwy klas
        save_path: Ścieżka do zapisania wykresu (opcjonalnie)
    """
    x = np.arange(len(class_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars1 = ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
    bars2 = ax.bar(x, recall, width, label='Recall', alpha=0.8)
    bars3 = ax.bar(x + width, f1, width, label='F1-Score', alpha=0.8)
    
    ax.set_xlabel('Klasa', fontsize=12)
    ax.set_ylabel('Wartość', fontsize=12)
    ax.set_title('Metryki per klasa', fontsize=16, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.1])
    
    # Dodaj wartości na słupkach
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=8
            )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Metryki per klasa zapisane w: {save_path}")
    
    plt.show()


def print_evaluation_report(results, class_names):
    """
    Wyświetla szczegółowy raport ewaluacji
    
    Args:
        results: Wyniki ewaluacji
        class_names: Nazwy klas
    """
    print("=" * 80)
    print("RAPORT EWALUACJI MODELU")
    print("=" * 80)
    
    print(f"\nOgólna dokładność (Accuracy): {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print(f"\nŚrednie ważone metryki:")
    print(f"  Precision: {results['precision_weighted']:.4f}")
    print(f"  Recall:    {results['recall_weighted']:.4f}")
    print(f"  F1-Score:  {results['f1_weighted']:.4f}")
    
    print("\n" + "-" * 80)
    print("METRYKI PER KLASA")
    print("-" * 80)
    print(f"{'Klasa':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-" * 80)
    
    for i, class_name in enumerate(class_names):
        print(
            f"{class_name:<20} "
            f"{results['precision'][i]:<12.4f} "
            f"{results['recall'][i]:<12.4f} "
            f"{results['f1'][i]:<12.4f} "
            f"{int(results['support'][i]):<10}"
        )
    
    print("\n" + "-" * 80)
    print("RAPORT KLASYFIKACJI (sklearn)")
    print("-" * 80)
    print(classification_report(
        results['y_true'],
        results['y_pred'],
        target_names=class_names,
        digits=4
    ))


def main():
    """
    Główna funkcja skryptu ewaluacji
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Ewaluuj wytrenowany model na danych testowych'
    )
    parser.add_argument(
        '--test-dir',
        type=str,
        default=None,
        help='Katalog z danymi testowymi (domyślnie: data/test)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Ścieżka do modelu (domyślnie: models/object_classifier_model.h5)'
    )
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Nie wyświetlaj wykresów'
    )
    
    args = parser.parse_args()
    
    # Określ ścieżki
    test_dir = args.test_dir if args.test_dir else config.TEST_DIR
    model_path = args.model if args.model else config.MODEL_PATH
    
    # Sprawdź czy istnieją dane testowe
    if not os.path.exists(test_dir):
        print(f"Ostrzeżenie: Katalog z danymi testowymi nie istnieje: {test_dir}")
        print("Jeśli chcesz ewaluować model, umieść dane testowe w tym katalogu.")
        return
    
    # Załaduj model
    print("Ładowanie modelu...")
    try:
        model = load_model(model_path)
        print("Model załadowany pomyślnie.\n")
    except FileNotFoundError as e:
        print(f"Błąd: {e}")
        print("Upewnij się, że model został wytrenowany. Uruchom najpierw train.py")
        return
    
    # Przygotuj generator danych testowych
    print("Przygotowywanie danych testowych...")
    test_generator = get_test_data_generator(test_dir, batch_size=config.BATCH_SIZE)
    print(f"Liczba próbek testowych: {test_generator.samples}")
    print(f"Liczba klas: {test_generator.num_classes}\n")
    
    # Wykonaj ewaluację
    results = evaluate_model(model, test_generator)
    
    # Wyświetl raport
    print_evaluation_report(results, config.CLASSES)
    
    # Wizualizacje
    if not args.no_plots:
        print("\nTworzenie wizualizacji...")
        
        # Macierz pomyłek
        cm_path = os.path.join(config.MODELS_DIR, 'confusion_matrix.png')
        plot_confusion_matrix(results['confusion_matrix'], config.CLASSES, save_path=cm_path)
        
        # Metryki per klasa
        metrics_path = os.path.join(config.MODELS_DIR, 'class_metrics.png')
        plot_class_metrics(
            results['precision'],
            results['recall'],
            results['f1'],
            config.CLASSES,
            save_path=metrics_path
        )


if __name__ == "__main__":
    main()

