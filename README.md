# Red Box Detection Tool

Automatyczne i interaktywne narzędzie do detekcji czerwonych ramek na arkuszach dokumentów, wzbogacone o uczenie maszynowe YOLOv8 — model uczy się na bieżąco wraz z każdym zatwierdzonym obrazem.

## Opis Projektu

System został opracowany w celu automatyzacji procesu lokalizacji i wycinania zaznaczonych na czerwono obszarów (np. pól podpisów lub pieczątek) ze skanów dokumentów. Projekt łączy algorytmy wizji komputerowej oparte na bibliotece OpenCV z interaktywnym interfejsem graficznym oraz modelem YOLOv8, który przyrostowo uczy się na zatwierdzonych danych i podpowiada kolejne ramki automatycznie.

## Kluczowe Funkcjonalności

- **Wydajna Detekcja HSV**: Wykorzystanie przestrzeni barw HSV, operacji morfologicznych oraz `RETR_LIST` do wykrywania nawet najdrobniejszych i zagnieżdżonych elementów.
- **Inteligentna Filtracja**: Automatyczne usuwanie kontenerów nadrzędnych, jeśli zawierają w sobie mniejsze docelowe obiekty.
- **Batch Processing**: Równoległe przetwarzanie zestawów obrazów, pozwalające na analizę wielu dokumentów jednocześnie.
- **Uczenie przyrostowe YOLOv8**: Po każdym zatwierdzonym obrazie model fine-tunuje się w tle i od 3. próbki zaczyna podpowiadać ramki na kolejnych obrazach jako żółte przerywane ramki.
- **Interaktywny Reviewer** z pełną obsługą podpowiedzi YOLO:
  - Żółte przerywane ramki — niezatwierdzone podpowiedzi YOLO.
  - `A` — zatwierdź wszystkie podpowiedzi YOLO naraz.
  - `R` — odrzuć wszystkie podpowiedzi YOLO.
  - `Ctrl+V` — wklej etykietę ze schowka (pełna obsługa polskich znaków).
  - Ręczne dodawanie, usuwanie i edycja etykiet ramek.
  - Szybka nawigacja po liście plików (`L`) z Page Up/Down, Home/End oraz skokiem do numeru.
- **Ustrukturyzowany Eksport**:
  - Przed eksportem automatycznie usuwa z `approved.json` ramki z pustą etykietą oraz niezatwierdzone podpowiedzi YOLO (etykiety `yolo:XX`).
  - Wszystkie wycinki zapisywane płasko w jednym folderze (`output/`).
  - Sekwencyjne nazewnictwo: `nazwapliku_f001.jpg`, `nazwapliku_f002.jpg`...
  - Raport `summary.csv` w formacie UTF-8 (bez nagłówka, bez BOM): `plik,etykieta`.
- **Normalizacja wycinków**:
  - `normalize_crops.py` — skaluje wycinki do jednolitego rozmiaru (bez zachowania proporcji).
  - `normalize_original_size.py` — kopiuje wycinki bez zmiany rozmiaru.

## Wymagania Systemowe

- Python 3.10 lub nowszy.
- Biblioteki:
  ```bash
  pip install opencv-python numpy pillow ultralytics pyperclip
  ```

## Struktura Projektu

```
projekt/
├── img/                               # Zdjęcia wejściowe
├── output/                            # Wycinki + summary.csv (po eksporcie)
├── outputNormalized/                  # Wycinki przeskalowane do jednolitego rozmiaru
├── OutputNormalized_Without_Reshape/  # Wycinki oryginalne (bez skalowania)
├── yolo_dataset/                      # Dane treningowe YOLO (tworzone automatycznie)
├── yolo_weights/                      # Wagi modelu YOLO (tworzone automatycznie)
├── approved.json                      # Wyniki detekcji i zatwierdzeń
├── extremes.json                      # Maksymalne wymiary wycinków
├── review.py                          # Główne narzędzie przeglądu i eksportu
├── detect.py                          # Masowa detekcja HSV (batch)
├── find_extremes.py                   # Szukanie największych wycinków
├── normalize_crops.py                 # Skalowanie wycinków do jednolitego rozmiaru
├── normalize_original_size.py         # Kopia wycinków bez skalowania
└── red_box_tool/
    ├── core/
    │   ├── config.py                  # Parametry detekcji
    │   ├── models.py                  # Klasa Box
    │   ├── geometry.py                # Funkcje IoU
    │   └── io_utils.py                # Odczyt/zapis plików
    ├── detection/
    │   ├── engine.py                  # Silnik detekcji HSV
    │   └── yolo_trainer.py            # Integracja YOLOv8
    └── interface/
        ├── reviewer.py                # GUI przeglądu ramek
        └── exporter.py                # Eksport wycinków i CSV
```

## Instrukcja Obsługi

### 1. Instalacja

```bash
pip install opencv-python numpy pillow ultralytics pyperclip
```

### 2. Przegląd i weryfikacja

```bash
python review.py
```

Jeśli plik `approved.json` nie istnieje, skrypt automatycznie uruchomi detekcję HSV. Po zatwierdzeniu każdego obrazu model YOLO trenuje się w tle i stopniowo poprawia jakość podpowiedzi.

Dodatkowe opcje:

```bash
python review.py --no-yolo     # Tryb klasyczny bez YOLO
python review.py --image ścieżka/do/pliku.jpg  # Jeden konkretny obraz
python review.py --reset       # Usuń approved.json i zacznij od nowa
```

### 3. Skróty klawiszowe w GUI

| Klawisz | Akcja |
|---|---|
| `Enter` (bez zaznaczenia) | Zapisz i przejdź do następnego |
| `L` | Lista plików z nawigacją |
| `A` | Zatwierdź wszystkie podpowiedzi YOLO |
| `R` | Odrzuć wszystkie podpowiedzi YOLO |
| `Enter` (przy zaznaczeniu) | Zatwierdź ramkę / podpowiedź YOLO |
| `Del` | Usuń zaznaczoną ramkę |
| `Backspace` | Usuń ostatni znak etykiety |
| `Ctrl+V` | Wklej etykietę ze schowka |
| `1` / `2` | Masowe OK / Odrzuć wszystkie ramki |
| `F` | Dopasuj widok do okna |
| `0` | Zoom 100% |
| `Scroll` | Zoom in/out |
| `Drag LPM` | Przesuwanie obrazu |
| `PPM` (drag) | Narysuj nową ramkę |
| `Q` / `Esc` | Wyjdź bez zapisu |

**Nawigacja w liście plików (`L`):**

| Klawisz | Akcja |
|---|---|
| `↑` / `↓` | O 1 pozycję |
| `Page Up` / `Page Down` | O 10 pozycji |
| `Home` / `End` | Pierwszy / ostatni plik |
| Cyfry + `Enter` | Skok do numeru pliku |

### 4. Ręczna detekcja wsadowa

```bash
python detect.py
```

### 5. Eksport wycinków

```bash
python review.py --extract
```

Przed eksportem program automatycznie:
1. Usuwa z `approved.json` ramki z pustą etykietą.
2. Usuwa niezatwierdzone podpowiedzi YOLO (etykiety `yolo:XX`).
3. Zapisuje oczyszczony `approved.json`.
4. Eksportuje wycinki do `output/` jako `nazwapliku_f001.jpg` wraz z `summary.csv`.

### 6. Normalizacja rozmiaru

```bash
# Znajdź maksymalne wymiary wycinków
python find_extremes.py

# Skaluj wszystkie wycinki do jednolitego rozmiaru (bez zachowania proporcji)
python normalize_crops.py

# Lub skopiuj wycinki bez zmiany rozmiaru
python normalize_original_size.py
```

## Konfiguracja

### Parametry detekcji HSV (`red_box_tool/core/config.py`)

| Parametr | Opis |
|---|---|
| `HSV_LOWER1/2`, `HSV_UPPER1/2` | Zakresy koloru czerwonego w HSV |
| `MIN_BOX_W/H` | Minimalne wymiary wykrywanych ramek |
| `SAVE_DEBUG_IMAGES` | Zapis masek diagnostycznych do `debug_tuning/` |

### Parametry YOLO (`red_box_tool/detection/yolo_trainer.py`)

| Parametr | Opis |
|---|---|
| `MIN_IMAGES_TO_TRAIN` | Minimalna liczba próbek do pierwszego treningu (domyślnie: 3) |
| `CONFIDENCE_THRESHOLD` | Minimalny próg pewności podpowiedzi (domyślnie: 0.35) |
| `BASE_MODEL` | Bazowy model YOLO (domyślnie: `yolov8n.pt`) |
