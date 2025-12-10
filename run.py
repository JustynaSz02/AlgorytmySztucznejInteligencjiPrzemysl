import os
import cv2
import numpy as np
import threading
import time
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ============================================================================
# STAŁE
# ============================================================================
MODEL_PATH = "models/shapes_cnn_mse.h5"
TRAIN_DIR = "data/train"
IMG_SIZE = (128, 128)  # Rozmiar obrazów wejściowych (szerokość x wysokość)
CAMERA_INDEX = 0  # Indeks kamery (domyślnie 0)


# ============================================================================
# WCZYTANIE MODELU I PRZYGOTOWANIE KLAS
# ============================================================================
def load_model_and_classes():
    """
    Wczytuje model CNN i automatycznie wykrywa nazwy klas z katalogów treningowych.
    Zwraca model i słownik mapujący indeksy do nazw klas.
    """
    # Sprawdzenie czy model istnieje
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model nie został znaleziony: {MODEL_PATH}")
    
    # Wczytanie modelu
    print(f"Wczytywanie modelu z: {MODEL_PATH}")
    model = load_model(MODEL_PATH)
    print("Model wczytany pomyślnie!")
    
    # Automatyczne wykrycie klas używając ImageDataGenerator (tak jak w train.py)
    datagen = ImageDataGenerator(rescale=1./255)
    train_generator = datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=1,
        class_mode='categorical',
        shuffle=False
    )
    
    # Mapowanie indeksów do nazw klas
    # class_indices to słownik: {nazwa_klasy: indeks}
    # Tworzymy odwrotne mapowanie: {indeks: nazwa_klasy}
    class_indices = train_generator.class_indices
    index_to_class = {v: k for k, v in class_indices.items()}
    
    print(f"Wykryto {len(index_to_class)} klas: {list(class_indices.keys())}")
    
    return model, index_to_class


# ============================================================================
# PRZETWARZANIE OBRAZU
# ============================================================================
def preprocess_frame(frame):
    """
    Przetwarza klatkę z kamery do formatu wymaganego przez model.
    
    Args:
        frame: Obraz BGR z kamery (OpenCV format)
    
    Returns:
        Przetworzony obraz gotowy do predykcji: (1, 128, 128, 3) RGB znormalizowany
    """
    # Zmiana rozmiaru do 128x128
    resized = cv2.resize(frame, IMG_SIZE)
    
    # Konwersja BGR → RGB (OpenCV używa BGR, model oczekuje RGB)
    rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    
    # Normalizacja pikseli: 0-255 → 0-1
    normalized = rgb_frame.astype(np.float32) / 255.0
    
    # Rozszerzenie wymiarów do (1, 128, 128, 3) dla modelu
    # Model oczekuje batch_size jako pierwszy wymiar
    processed = np.expand_dims(normalized, axis=0)
    
    return processed


# ============================================================================
# TWORZENIE PANELU Z PREDYKCJAMI
# ============================================================================
def create_prediction_panel(predictions, index_to_class, prediction_fps=None, panel_width=400, panel_height=720):
    """
    Tworzy panel z rankingiem predykcji (wszystkie klasy w kolejności malejącej).
    
    Args:
        predictions: Tablica prawdopodobieństw z modelu (shape: (num_classes,))
        index_to_class: Słownik mapujący indeksy do nazw klas
        prediction_fps: Liczba predykcji na sekundę (opcjonalne)
        panel_width: Szerokość panelu w pikselach
        panel_height: Wysokość panelu w pikselach
    
    Returns:
        Obraz panelu jako numpy array (BGR dla OpenCV)
    """
    # Utworzenie białego panelu
    panel = np.ones((panel_height, panel_width, 3), dtype=np.uint8) * 255
    
    # Sortowanie predykcji w kolejności malejącej
    # predictions to tablica prawdopodobieństw dla każdej klasy
    sorted_indices = np.argsort(predictions)[::-1]  # Indeksy posortowane malejąco
    
    # Nagłówek
    cv2.putText(panel, "PREDYKCJE:", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
    
    # Wyświetlanie FPS predykcji
    if prediction_fps is not None:
        fps_text = f"FPS: {prediction_fps:.1f}"
        cv2.putText(panel, fps_text, (20, 75), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 150, 0), 2)
    
    # Wyświetlanie wszystkich klas w kolejności malejącej
    y_offset = 110  # Zwiększone o 20 pikseli ze względu na FPS
    line_height = 50
    
    for rank, class_idx in enumerate(sorted_indices, start=1):
        class_name = index_to_class[class_idx]
        probability = predictions[class_idx]
        percentage = probability * 100
        
        # Format: "1. blue_circle: 95.2%"
        text = f"{rank}. {class_name}: {percentage:.1f}%"
        
        # Kolor tekstu - ciemniejszy dla wyższych prawdopodobieństw
        color_intensity = int(255 * (1 - probability * 0.5))  # Im wyższe prawdopodobieństwo, tym ciemniejszy tekst
        color = (color_intensity, color_intensity, color_intensity)
        
        # Rysowanie tekstu
        cv2.putText(panel, text, (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Pasek postępu wizualizujący prawdopodobieństwo
        bar_width = int((panel_width - 60) * probability)
        cv2.rectangle(panel, (20, y_offset + 10), (20 + bar_width, y_offset + 25), 
                     (100, 150, 200), -1)
        
        y_offset += line_height
    
    return panel


# ============================================================================
# FUNKCJA GŁÓWNA
# ============================================================================
def main():
    """
    Główna funkcja aplikacji: inicjalizuje kamerę, wczytuje model,
    i uruchamia pętlę przetwarzania obrazu z kamery w czasie rzeczywistym.
    """
    try:
        # Wczytanie modelu i klas
        model, index_to_class = load_model_and_classes()
        
        # Inicjalizacja kamery
        print(f"Inicjalizacja kamery (indeks {CAMERA_INDEX})...")
        cap = cv2.VideoCapture(CAMERA_INDEX)
        
        if not cap.isOpened():
            raise RuntimeError(f"Nie można otworzyć kamery o indeksie {CAMERA_INDEX}")
        
        print("Kamera zainicjalizowana pomyślnie!")
        print("\nInstrukcje:")
        print("  - Naciśnij 'q' lub ESC aby zamknąć aplikację")
        print("  - Zamknij okno aby zakończyć program\n")
        
        # Utworzenie okna z możliwością zamknięcia przez przycisk
        window_name = "Kamera + Predykcja Modelu"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        # Zmienne do mechanizmu pomijania klatek
        processing_lock = threading.Lock()
        processing = False
        frame_to_process = None
        current_predictions = None
        current_display_frame = None
        
        # Zmienne do licznika FPS predykcji
        prediction_fps = 0.0
        last_prediction_time = None
        fps_alpha = 0.1  # Współczynnik wygładzania FPS (exponential moving average)
        
        def process_frame_async():
            """Funkcja przetwarzająca klatkę w osobnym wątku"""
            nonlocal processing, current_predictions, frame_to_process, prediction_fps, last_prediction_time
            
            try:
                # Zapisanie czasu rozpoczęcia predykcji
                prediction_start_time = time.time()
                
                # Przetwarzanie obrazu
                processed_frame = preprocess_frame(frame_to_process)
                
                # Predykcja modelu
                predictions = model.predict(processed_frame, verbose=0)[0]  # [0] bo model zwraca batch
                
                # Obliczenie czasu predykcji i FPS
                prediction_end_time = time.time()
                prediction_duration = prediction_end_time - prediction_start_time
                
                if prediction_duration > 0:
                    current_fps = 1.0 / prediction_duration
                    
                    # Obliczenie wygładzonego FPS (exponential moving average)
                    with processing_lock:
                        if last_prediction_time is None:
                            prediction_fps = current_fps
                        else:
                            # Wygladzanie FPS używając exponential moving average
                            prediction_fps = fps_alpha * current_fps + (1 - fps_alpha) * prediction_fps
                        last_prediction_time = prediction_end_time
                
                # Zapisanie wyników
                with processing_lock:
                    current_predictions = predictions
                    processing = False
            except Exception as e:
                print(f"Błąd podczas przetwarzania: {e}")
                with processing_lock:
                    processing = False
        
        # Pętla główna
        while True:
            # Przechwycenie klatki z kamery (zawsze czytamy najnowszą klatkę)
            ret, frame = cap.read()
            
            if not ret:
                print("Błąd: Nie można przechwycić klatki z kamery")
                break
            
            # Zapisanie klatki do wyświetlenia
            current_display_frame = frame.copy()
            
            # Mechanizm pomijania klatek podczas przetwarzania
            with processing_lock:
                if not processing:
                    # Zapisanie aktualnej klatki do przetworzenia
                    frame_to_process = frame.copy()
                    processing = True
                    
                    # Uruchomienie przetwarzania w osobnym wątku
                    thread = threading.Thread(target=process_frame_async)
                    thread.daemon = True
                    thread.start()
            # Jeśli processing == True, po prostu pomijamy tę klatkę i czytamy następną
            
            # Wyświetlanie (używamy najnowszej klatki i najnowszych predykcji)
            if current_predictions is not None and current_display_frame is not None:
                # Pobranie aktualnego FPS (thread-safe)
                with processing_lock:
                    current_fps = prediction_fps
                
                # Utworzenie panelu z predykcjami
                prediction_panel = create_prediction_panel(
                    current_predictions, 
                    index_to_class,
                    prediction_fps=current_fps,
                    panel_width=400,
                    panel_height=current_display_frame.shape[0]  # Wysokość panelu = wysokość klatki
                )
                
                # Połączenie obrazu kamery i panelu predykcji
                # Sprawdzenie czy wysokości się zgadzają
                if current_display_frame.shape[0] != prediction_panel.shape[0]:
                    # Dopasowanie wysokości panelu do klatki
                    prediction_panel = cv2.resize(
                        prediction_panel, 
                        (prediction_panel.shape[1], current_display_frame.shape[0])
                    )
                
                # Połączenie poziome (obraz kamery | panel predykcji)
                combined_image = np.hstack([current_display_frame, prediction_panel])
                
                # Wyświetlanie połączonego obrazu
                cv2.imshow(window_name, combined_image)
            elif current_display_frame is not None:
                # Jeśli jeszcze nie ma predykcji, wyświetl tylko obraz z kamery
                cv2.imshow(window_name, current_display_frame)
            
            # Obsługa klawiszy
            key = cv2.waitKey(1) & 0xFF
            
            # Zamknięcie gdy ESC (27) lub 'q'
            if key == 27 or key == ord('q'):
                print("Zamykanie aplikacji...")
                break
            
            # Sprawdzenie czy okno zostało zamknięte przez użytkownika
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                print("Okno zamknięte przez użytkownika")
                break
        
    except FileNotFoundError as e:
        print(f"Błąd: {e}")
        print("Upewnij się, że model został wytrenowany (uruchom train.py)")
    except RuntimeError as e:
        print(f"Błąd: {e}")
        print("Sprawdź czy kamera jest podłączona i dostępna")
    except Exception as e:
        print(f"Nieoczekiwany błąd: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Zwolnienie zasobów
        if 'cap' in locals():
            cap.release()
        cv2.destroyAllWindows()
        print("Zasoby zwolnione. Do widzenia!")


# ============================================================================
# URUCHOMIENIE APLIKACJI
# ============================================================================
if __name__ == "__main__":
    main()

