"""
Skrypt do wykonywania predykcji na nowych obrazach
"""
import os
import sys
import argparse
import numpy as np
from PIL import Image
# Dodaj główny katalog do ścieżki Python
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

import config
from src.model import load_model
from src.data_loader import preprocess_image


def predict_image(model, image_path, top_k=3):
    """
    Wykonuje predykcję na pojedynczym obrazie
    
    Args:
        model: Wytrenowany model Keras
        image_path: Ścieżka do obrazu
        top_k: Liczba najlepszych predykcji do wyświetlenia
    
    Returns:
        Słownik z predykcjami i prawdopodobieństwami
    """
    # Przetwórz obraz
    img_array = preprocess_image(image_path)
    
    # Wykonaj predykcję
    predictions = model.predict(img_array, verbose=0)
    
    # Pobierz top_k najlepszych predykcji
    top_indices = np.argsort(predictions[0])[-top_k:][::-1]
    
    results = []
    for idx in top_indices:
        results.append({
            'class': config.CLASSES[idx],
            'probability': float(predictions[0][idx])
        })
    
    return results


def predict_batch(model, image_dir):
    """
    Wykonuje predykcje na wszystkich obrazach w katalogu
    
    Args:
        model: Wytrenowany model Keras
        image_dir: Ścieżka do katalogu z obrazami
    
    Returns:
        Lista wyników dla każdego obrazu
    """
    results = []
    
    # Obsługiwane formaty obrazów
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
    
    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"Katalog nie istnieje: {image_dir}")
    
    # Pobierz listę plików obrazów
    image_files = [
        f for f in os.listdir(image_dir)
        if f.lower().endswith(image_extensions)
    ]
    
    if not image_files:
        print(f"Nie znaleziono obrazów w katalogu: {image_dir}")
        return results
    
    print(f"Znaleziono {len(image_files)} obrazów. Wykonywanie predykcji...\n")
    
    for filename in image_files:
        image_path = os.path.join(image_dir, filename)
        try:
            pred_results = predict_image(model, image_path, top_k=1)
            best_pred = pred_results[0]
            results.append({
                'filename': filename,
                'predicted_class': best_pred['class'],
                'confidence': best_pred['probability']
            })
            
            print(f"{filename}:")
            print(f"  Przewidywana klasa: {best_pred['class']}")
            print(f"  Prawdopodobieństwo: {best_pred['probability']:.4f}")
            print()
            
        except Exception as e:
            print(f"Błąd przy przetwarzaniu {filename}: {e}")
            continue
    
    return results


def display_prediction(image_path, results):
    """
    Wyświetla szczegółowe wyniki predykcji
    
    Args:
        image_path: Ścieżka do obrazu
        results: Wyniki predykcji
    """
    print("=" * 60)
    print(f"Obraz: {os.path.basename(image_path)}")
    print("=" * 60)
    print("\nPredykcje (top 3):")
    print("-" * 60)
    
    for i, result in enumerate(results, 1):
        percentage = result['probability'] * 100
        bar_length = int(percentage / 2)  # Długość paska na wykresie
        bar = '█' * bar_length + '░' * (50 - bar_length)
        
        print(f"{i}. {result['class']:20s} | {percentage:5.2f}% | {bar}")
    
    print("=" * 60)
    
    # Pokaż najlepszą predykcję
    best = results[0]
    print(f"\nNajlepsza predykcja: {best['class']} ({best['probability']*100:.2f}%)")
    print()


def main():
    """
    Główna funkcja skryptu predykcji
    """
    parser = argparse.ArgumentParser(
        description='Wykonaj predykcję na obrazach używając wytrenowanego modelu'
    )
    parser.add_argument(
        '--image',
        type=str,
        help='Ścieżka do pojedynczego obrazu'
    )
    parser.add_argument(
        '--dir',
        type=str,
        help='Ścieżka do katalogu z obrazami'
    )
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Ścieżka do modelu (domyślnie: models/object_classifier_model.h5)'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=3,
        help='Liczba najlepszych predykcji do wyświetlenia (domyślnie: 3)'
    )
    
    args = parser.parse_args()
    
    # Sprawdź argumenty
    if not args.image and not args.dir:
        parser.error("Musisz podać --image lub --dir")
    
    if args.image and args.dir:
        parser.error("Możesz podać tylko --image lub --dir, nie oba")
    
    # Załaduj model
    print("Ładowanie modelu...")
    try:
        if args.model:
            model = load_model(args.model)
        else:
            model = load_model()
        print("Model załadowany pomyślnie.\n")
    except FileNotFoundError as e:
        print(f"Błąd: {e}")
        print("Upewnij się, że model został wytrenowany. Uruchom najpierw train.py")
        return
    
    # Wykonaj predykcje
    if args.image:
        # Predykcja dla pojedynczego obrazu
        if not os.path.exists(args.image):
            print(f"Błąd: Obraz nie istnieje: {args.image}")
            return
        
        results = predict_image(model, args.image, top_k=args.top_k)
        display_prediction(args.image, results)
    
    elif args.dir:
        # Predykcje dla katalogu
        results = predict_batch(model, args.dir)
        
        # Podsumowanie
        if results:
            print("\n" + "=" * 60)
            print("PODSUMOWANIE")
            print("=" * 60)
            print(f"Przetworzono: {len(results)} obrazów")
            
            # Statystyki klas
            class_counts = {}
            for result in results:
                cls = result['predicted_class']
                class_counts[cls] = class_counts.get(cls, 0) + 1
            
            print("\nRozkład przewidywanych klas:")
            for cls, count in sorted(class_counts.items()):
                percentage = (count / len(results)) * 100
                print(f"  {cls:20s}: {count:3d} ({percentage:5.2f}%)")


if __name__ == "__main__":
    main()

