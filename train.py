import os
import csv
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # Generator danych obrazowych z augmentacją
from tensorflow.keras.models import Sequential  # Model sekwencyjny (warstwa po warstwie)
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout  # Warstwy sieci neuronowej
from tensorflow.keras.optimizers import Adam  # Optymalizator Adam (adaptacyjny learning rate)


BASE_DATA_DIR = "data"                
TRAIN_DIR = os.path.join(BASE_DATA_DIR, "train")  # Katalog z danymi treningowymi
VAL_DIR   = os.path.join(BASE_DATA_DIR, "val")     # Katalog z danymi walidacyjnymi
TEST_DIR  = os.path.join(BASE_DATA_DIR, "test")    # Katalog z danymi testowymi

IMG_SIZE = (128, 128)      # Rozmiar obrazów wejściowych (szerokość x wysokość)
BATCH_SIZE = 32            # Liczba próbek przetwarzanych jednocześnie (większy batch = szybsze, ale więcej pamięci)
LEARNING_RATE = 1e-3       # Współczynnik uczenia (0.001) - jak szybko model się uczy
MIN_ACCURACY = 0.90         # Minimalna wymagana dokładność na zbiorze walidacyjnym (90%)
MAX_EPOCHS = 100           # Maksymalna liczba epok (zabezpieczenie przed nieskończoną pętlą)

datagen = ImageDataGenerator(rescale=1./255)  # Normalizacja pikseli: 0-255 -> 0-1

train_generator = datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

val_generator = datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

test_generator = datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Automatyczne wykrycie liczby klas na podstawie liczby podkatalogów
num_classes = train_generator.num_classes
print("Klasy:", train_generator.class_indices)  # Wyświetla mapowanie: nazwa klasy -> indeks



# ============================================================================
# BUDOWA ARCHITEKTURY MODELU CNN (CONVOLUTIONAL NEURAL NETWORK)
# ============================================================================
# CNN składa się z warstw konwolucyjnych (wykrywają wzorce) i warstw gęstych (klasyfikacja)
model = Sequential([
    # WARSTWA 1: Konwolucyjna - wykrywa proste wzorce (krawędzie, linie)
    # 32 filtry 3x3, każdy szuka różnych wzorców w obrazie
    Conv2D(32, (3, 3), activation='relu',
           input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),  # Wejście: 128x128x3 (RGB)
    MaxPooling2D((2, 2)),  # Redukcja wymiarów 2x (128x128 -> 64x64) - zmniejsza liczbę parametrów
    
    # WARSTWA 2: Konwolucyjna - wykrywa bardziej złożone wzorce (kształty)
    # 64 filtry - więcej filtrów = więcej wykrywanych wzorców
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),  # Redukcja: 64x64 -> 32x32
    
    # WARSTWA 3: Konwolucyjna - wykrywa bardzo złożone wzorce (całe obiekty)
    # 128 filtrów - najwięcej filtrów w ostatniej warstwie konwolucyjnej
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),  # Redukcja: 32x32 -> 16x16
    
    # Przekształcenie 2D -> 1D (spłaszczenie) - przygotowanie do warstw gęstych
    Flatten(),
    
    # WARSTWA GĘSTA 1: Warstwa ukryta - uczy złożone kombinacje cech
    Dense(128, activation='relu'),  # 128 neuronów z aktywacją ReLU
    Dropout(0.3),  # Wyłącza losowo 30% neuronów podczas treningu (zapobiega przeuczeniu)
    
    # WARSTWA WYJŚCIOWA: Klasyfikacja - zwraca prawdopodobieństwa dla każdej klasy
    Dense(num_classes, activation='softmax')  # Softmax - suma prawdopodobieństw = 1.0
])

# ============================================================================
# KOMPILACJA MODELU
# ============================================================================
# Kompilacja definiuje jak model będzie się uczył i jak będzie oceniany

model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),  # Optymalizator Adam - adaptacyjnie dostosowuje learning rate
    loss='categorical_crossentropy',              # Funkcja straty dla klasyfikacji wieloklasowej
    metrics=['accuracy', 'mse']                   # Metryki do monitorowania:
                                                  # - accuracy: procent poprawnie sklasyfikowanych próbek
                                                  # - mse: błąd średniokwadratowy (Mean Squared Error)
)

model.summary() # Wyświetlenie struktury modelu - pokazuje liczbę parametrów w każdej warstwie



# ============================================================================
# TRENING MODELU
# ============================================================================
# Trening do osiągnięcia MIN_ACCURACY lub MAX_EPOCHS

print(f"Trening do osiągnięcia {MIN_ACCURACY*100:.0f}% dokładności (max {MAX_EPOCHS} epok)\n")

history = {'loss': [], 'accuracy': [], 'mse': [], 'val_loss': [], 'val_accuracy': [], 'val_mse': []}
current_epoch = 0
val_accuracy = 0.0

while val_accuracy < MIN_ACCURACY and current_epoch < MAX_EPOCHS:
    current_epoch += 1
    epoch_history = model.fit(
        train_generator,
        epochs=1,
        validation_data=val_generator,
        verbose=1
    )
    
    # Aktualizacja historii
    for key in history.keys():
        history[key].extend(epoch_history.history[key])
    
    val_accuracy = history['val_accuracy'][-1]
    print(f"Epoka {current_epoch}: Dokładność walidacyjna = {val_accuracy*100:.2f}%\n")

if val_accuracy >= MIN_ACCURACY:
    print(f"✓ Osiągnięto wymaganą dokładność ({val_accuracy*100:.2f}%) po {current_epoch} epokach\n")
else:
    print(f"⚠ Osiągnięto maksymalną liczbę epok ({MAX_EPOCHS}). Aktualna dokładność: {val_accuracy*100:.2f}%\n")



# ============================================================================
# OCENA MODELU NA ZBIORZE TESTOWYM
# ============================================================================
# evaluate() - ocenia model na danych testowych (które model NIE widział podczas treningu)
# To jest ostateczna ocena wydajności modelu

test_loss, test_acc, test_mse = model.evaluate(test_generator, verbose=1)
print(f"\nWyniki na TEŚCIE:")
print(f"  loss = {test_loss:.4f}")        # Wartość funkcji straty (niższa = lepsza)
print(f"  accuracy = {test_acc:.4f}")     # Dokładność (wyższa = lepsza, max 1.0 = 100%)
print(f"  MSE = {test_mse:.4f}")          # Błąd średniokwadratowy (niższy = lepszy)



# ============================================================================
# WIZUALIZACJA WYNIKÓW TRENINGU
# ============================================================================
# Wykresy pokazują jak model uczył się w czasie (każda epoka)
# Porównanie train vs validation pomaga wykryć przeuczenie (overfitting)

epochs_range = range(1, len(history['loss']) + 1)  # Zakres epok do wykresów

plt.figure(figsize=(15, 4))  # Utworzenie figury z 3 wykresami obok siebie

# WYKRES 1: Dokładność (Accuracy)
plt.subplot(1, 3, 1)
plt.plot(epochs_range, history['accuracy'], label='train acc')      # Dokładność na danych treningowych
plt.plot(epochs_range, history['val_accuracy'], label='val acc')    # Dokładność na danych walidacyjnych
plt.title('Dokładność (accuracy)')
plt.xlabel('Epoka')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
# Interpretacja: Oba wykresy powinny rosnąć. Jeśli train >> val = przeuczenie

# WYKRES 2: Funkcja straty (Loss)
plt.subplot(1, 3, 2)
plt.plot(epochs_range, history['loss'], label='train loss')        # Strata na danych treningowych
plt.plot(epochs_range, history['val_loss'], label='val loss')      # Strata na danych walidacyjnych
plt.title('Funkcja kosztu (loss)')
plt.xlabel('Epoka')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
# Interpretacja: Oba wykresy powinny maleć. Jeśli val_loss rośnie = przeuczenie

# WYKRES 3: Błąd średniokwadratowy (MSE)
plt.subplot(1, 3, 3)
plt.plot(epochs_range, history['mse'], label='train MSE')          # MSE na danych treningowych
plt.plot(epochs_range, history['val_mse'], label='val MSE')        # MSE na danych walidacyjnych
plt.title('Błąd średniokwadratowy (MSE)')
plt.xlabel('Epoka')
plt.ylabel('MSE')
plt.legend()
plt.grid(True)
# Interpretacja: MSE mierzy średnią różnicę między przewidywaniami a prawdziwymi wartościami

plt.tight_layout()  # Automatyczne dopasowanie odstępów między wykresami

# Zapis wykresów do pliku PNG
os.makedirs("models", exist_ok=True)
plot_path = "models/training_history.png"
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"Wykresy zapisane w: {plot_path}")
plt.close()  # Zamknięcie figury, aby zwolnić pamięć



# ============================================================================
# ZAPIS DANYCH DO CSV
# ============================================================================
# Zapis historii treningu do pliku CSV

csv_path = "models/training_history.csv"
with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    
    # Nagłówki kolumn
    headers = ['epoch', 'loss', 'accuracy', 'mse', 'val_loss', 'val_accuracy', 'val_mse']
    writer.writerow(headers)
    
    # Dane z każdej epoki
    for epoch in range(len(history['loss'])):
        row = [
            epoch + 1,
            history['loss'][epoch],
            history['accuracy'][epoch],
            history['mse'][epoch],
            history['val_loss'][epoch],
            history['val_accuracy'][epoch],
            history['val_mse'][epoch]
        ]
        writer.writerow(row)

print(f"Dane treningu zapisane w: {csv_path}")


# ============================================================================
# ZAPIS WYTRENOWANEGO MODELU
# ============================================================================
# Zapis modelu do pliku .h5 (format HDF5) - można później wczytać bez ponownego treningu

os.makedirs("models", exist_ok=True)  # Utworzenie katalogu "models" jeśli nie istnieje

# Usuwanie poprzedniego modelu jeśli istnieje (nadpisywanie)
model_path = "models/shapes_cnn_mse.h5"
if os.path.exists(model_path):
    os.remove(model_path)
    print(f"Usunięto poprzedni model: {model_path}")

model.save(model_path)  # Zapis całego modelu (architektura + wagi)
print(f"Model zapisany w: {model_path}")
