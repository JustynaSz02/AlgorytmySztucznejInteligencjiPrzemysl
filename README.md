# System Rozpoznawania Obiektów z Użyciem Uczenia Maszynowego

System do klasyfikacji obiektów ze zdjęć z wykorzystaniem głębokiego uczenia (deep learning). System rozpoznaje 6 klas obiektów:

- **Czerwone koło** (`red_circle`)
- **Zielone koło** (`green_circle`)
- **Niebieskie koło** (`blue_circle`)
- **Czerwony kwadrat** (`red_square`)
- **Zielony kwadrat** (`green_square`)
- **Niebieski kwadrat** (`blue_square`)

## Technologie

- **Python 3.8+**
- **TensorFlow/Keras** - główna biblioteka do uczenia maszynowego
- **MobileNetV2** - pre-trenowany model do transfer learningu
- **NumPy, Pillow** - przetwarzanie obrazów
- **Matplotlib, Seaborn** - wizualizacja wyników
- **scikit-learn** - metryki i pomocnicze funkcje

## Struktura Projektu

```
BigDataPrzemysl/
├── data/
│   ├── train/              # Dane treningowe
│   │   ├── red_circle/
│   │   ├── green_circle/
│   │   ├── blue_circle/
│   │   ├── red_square/
│   │   ├── green_square/
│   │   └── blue_square/
│   └── test/               # Dane testowe (opcjonalnie)
├── models/                 # Wytrenowane modele
├── src/
│   ├── data_loader.py      # Ładowanie i przygotowanie danych
│   ├── model.py            # Architektura modelu
│   ├── train.py            # Skrypt treningowy
│   ├── predict.py          # Skrypt do predykcji
│   └── evaluate.py         # Ewaluacja modelu
├── config.py               # Konfiguracja systemu
├── requirements.txt        # Zależności Python
└── README.md               # Dokumentacja
```

## Instalacja

### 1. Klonowanie repozytorium

```bash
git clone <repo-url>
cd BigDataPrzemysl
```

### 2. Utworzenie środowiska wirtualnego

**Windows:**
```bash
python -m venv venv
.\venv\Scripts\Activate.ps1
```


### 3. Instalacja zależności

```bash
pip install -r requirements.txt
```

## Przygotowanie Danych

### Struktura katalogów

Przygotuj swoje dane treningowe zgodnie z następującą strukturą:

```
data/
└── train/
    ├── red_circle/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    ├── green_circle/
    │   └── ...
    ├── blue_circle/
    │   └── ...
    ├── red_square/
    │   └── ...
    ├── green_square/
    │   └── ...
    └── blue_square/
        └── ...
```

**Wymagania:**
- Każda klasa powinna mieć swój folder w `data/train/`
- W każdym folderze umieść co najmniej kilkadziesiąt zdjęć
- Obsługiwane formaty: `.png`, `.jpg`, `.jpeg`, `.bmp`, `.gif`
- Zalecane: różne orientacje i warunki oświetleniowe dla lepszej generalizacji

**Opcjonalnie - dane testowe:**

Jeśli chcesz ewaluować model, przygotuj podobną strukturę w `data/test/`:

```
data/
└── test/
    ├── red_circle/
    ├── green_circle/
    └── ...
```

## Użycie

### 1. Trening modelu

Aby wytrenować model na swoich danych:

```bash
python src/train.py
```

**Co się dzieje:**
- Dane są automatycznie podzielone na zbiór treningowy (80%) i walidacyjny (20%)
- Stosowana jest data augmentation (rotacja, przesunięcie, zmiana jasności)
- Model używa transfer learningu z MobileNetV2
- Najlepszy model jest zapisywany w `models/object_classifier_model.h5`
- Tworzone są wykresy historii treningu

**Parametry treningu** można zmienić w pliku `config.py`:
- `BATCH_SIZE` - rozmiar batcha (domyślnie: 32)
- `EPOCHS` - maksymalna liczba epok (domyślnie: 50)
- `LEARNING_RATE` - learning rate (domyślnie: 0.001)
- `VALIDATION_SPLIT` - proporcja danych walidacyjnych (domyślnie: 0.2)

### 2. Wykonywanie predykcji

#### Predykcja dla pojedynczego obrazu:

```bash
python src/predict.py --image ścieżka/do/obrazu.jpg
```

#### Predykcja dla wszystkich obrazów w katalogu:

```bash
python src/predict.py --dir ścieżka/do/katalogu
```

#### Dodatkowe opcje:

```bash
# Użyj innego modelu
python src/predict.py --image obraz.jpg --model models/inny_model.h5

# Wyświetl top 5 predykcji zamiast top 3
python src/predict.py --image obraz.jpg --top-k 5
```

### 3. Ewaluacja modelu

Aby ocenić dokładność modelu na danych testowych:

```bash
python src/evaluate.py
```

Lub z niestandardowymi ścieżkami:

```bash
python src/evaluate.py --test-dir data/test --model models/object_classifier_model.h5
```

**Wygenerowane raporty:**
- Dokładność ogólna (accuracy)
- Precision, Recall, F1-Score per klasa
- Macierz pomyłek (confusion matrix)
- Wykresy metryk per klasa

Wizualizacje są zapisywane w katalogu `models/`.

## Konfiguracja

Wszystkie parametry konfiguracyjne znajdują się w pliku `config.py`. Możesz tam zmienić:

- Ścieżki do danych i modeli
- Parametry treningu (batch size, epochs, learning rate)
- Parametry data augmentation
- Architekturę modelu bazowego
- Parametry callbacków (early stopping, reduce LR)

## Funkcje Data Augmentation

System automatycznie stosuje następujące przekształcenia podczas treningu:

- **Rotacja**: ±20 stopni
- **Przesunięcie**: ±10% w poziomie i pionie
- **Zmiana jasności**: zakres 80-120%
- **Odwrócenie poziome**: losowe
- **Zoom**: ±10%

Te przekształcenia pomagają modelowi lepiej generalizować na nowych danych.

## Architektura Modelu

Model używa **transfer learningu** z pre-trenowanego **MobileNetV2** (wytrenowanego na ImageNet):

1. **Warstwa bazowa**: MobileNetV2 (zamrożone warstwy, z wyjątkiem ostatnich 20)
2. **Global Average Pooling**: redukcja wymiarów
3. **Dropout** (0.5): regularyzacja
4. **Warstwa wyjściowa**: Dense layer z 6 neuronami (softmax)

## Callbacki Treningowe

- **EarlyStopping**: Zatrzymuje trening jeśli brak poprawy przez 10 epok
- **ModelCheckpoint**: Zapisuje najlepszy model (najniższy val_loss)
- **ReduceLROnPlateau**: Zmniejsza learning rate jeśli brak poprawy przez 5 epok
- **CSVLogger**: Zapisuje historię treningu do pliku CSV

## Przykłady Wyników

Po treningu otrzymujesz:
- Wytrenowany model: `models/object_classifier_model.h5`
- Historię treningu: `models/training_log.csv`
- Wykresy treningu: `models/training_history.png`
- (Po ewaluacji) Macierz pomyłek: `models/confusion_matrix.png`
- (Po ewaluacji) Metryki per klasa: `models/class_metrics.png`

## Rozwiązywanie Problemów

### Błąd: "Katalog z danymi treningowymi nie istnieje"
- Upewnij się, że utworzyłeś folder `data/train/` z podfolderami dla każdej klasy
- Sprawdź, czy nazwy folderów odpowiadają klasom w `config.py`

### Błąd: "Model nie został znaleziony"
- Najpierw wytrenuj model używając `python src/train.py`
- Sprawdź, czy model został zapisany w `models/object_classifier_model.h5`

### Niska dokładność modelu
- Zwiększ liczbę obrazów treningowych (co najmniej 50-100 per klasa)
- Sprawdź jakość i różnorodność danych
- Zwiększ liczbę epok lub zmień parametry data augmentation
- Rozważ fine-tuning większej liczby warstw bazowego modelu

### Problemy z pamięcią GPU
- Zmniejsz `BATCH_SIZE` w `config.py`
- Użyj `pretrained=False` w `model.py` (ale model będzie wolniejszy w uczeniu)

## Licencja

[Określ licencję]

## Autor

[Twoje dane]

