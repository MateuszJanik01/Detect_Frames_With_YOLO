# Instrukcja Obsługi: Red Box Tool

Narzędzie **Red Box Tool** służy do automatycznej detekcji, manualnej weryfikacji oraz ustrukturyzowanego eksportu czerwonych ramek (np. pól podpisów, pieczątek) z dokumentów graficznych. Narzędzie wyposażone jest w model YOLOv8, który uczy się na bieżąco i z każdym zatwierdzonym obrazem podpowiada kolejne ramki automatycznie.

---

## 1. Instalacja i Przygotowanie

### Wymagania

- **Python 3.10** lub nowszy.
- Biblioteki: `opencv-python`, `numpy`, `pillow`, `ultralytics`, `pyperclip`.

### Instalacja zależności

Otwórz terminal w folderze projektu i wykonaj:

```bash
pip install opencv-python numpy pillow ultralytics pyperclip
```

### Przygotowanie danych

1. Umieść zdjęcia dokumentów w formacie `.jpg`, `.png` lub `.bmp` w katalogu `img/`.
2. (Opcjonalnie) Jeśli masz listę oczekiwanej liczby ramek dla poszczególnych plików, wpisz ją do `correct_count_of_red_boxes.txt` w formacie: `nazwa_pliku.jpg;liczba`.

---

## 2. Przepływ Pracy (Workflow)

### KROK 1: Automatyczna Detekcja

Uruchom skrypt, aby szybko przetworzyć wszystkie zdjęcia w folderze `img/`:

```bash
python detect.py
```

- Skrypt wykorzystuje **wszystkie rdzenie procesora** (tryb równoległy).
- Wyniki zostaną zapisane w pliku `approved.json`.
- W folderze `debug_tuning/` pojawią się maski diagnostyczne (jeśli włączone w konfiguracji).

### KROK 2: Weryfikacja i Edycja (GUI)

Uruchom interfejs graficzny, aby sprawdzić poprawność wykrytych ramek:

```bash
python review.py
```

- Jeśli `approved.json` nie istnieje, skrypt automatycznie uruchomi detekcję HSV.
- Od 3. zatwierdzonego obrazu model YOLO zaczyna podpowiadać ramki jako **żółte przerywane ramki**.
- Możesz poprawiać błędy automatu, dodawać brakujące ramki i nadawać im etykiety.
- Etykiety z polskimi znakami wklejaj przez `Ctrl+V` (skopiuj tekst w dowolnym miejscu, wklej w GUI).

Dodatkowe opcje uruchomienia:

```bash
python review.py --no-yolo     # Tryb klasyczny bez YOLO
python review.py --image ścieżka/do/pliku.jpg  # Jeden konkretny obraz
python review.py --reset       # Usuń approved.json i zacznij od nowa
```

### KROK 3: Finalna Ekstrakcja

Gdy wszystkie ramki są zweryfikowane i oetykietowane, wyeksportuj je do plików:

```bash
python review.py --extract
```

Przed eksportem program automatycznie:
1. Usuwa z `approved.json` ramki z pustą etykietą.
2. Usuwa niezatwierdzone podpowiedzi YOLO (etykiety `yolo:XX`).
3. Zapisuje oczyszczony `approved.json`.
4. Eksportuje wycinki do `output/` jako `nazwapliku_f001.jpg` wraz z `summary.csv`.

### KROK 4: Normalizacja Rozmiaru (opcjonalnie)

```bash
# Znajdź maksymalne wymiary wycinków
python find_extremes.py

# Skaluj wszystkie wycinki do jednolitego rozmiaru (bez zachowania proporcji)
python normalize_crops.py

# Lub skopiuj wycinki bez zmiany rozmiaru
python normalize_original_size.py
```

---

## 3. Sterowanie w Interfejsie (review.py)

### Myszka

| Akcja | Opis |
|---|---|
| **LPM** (klik) | Zaznaczenie ramki |
| **LPM** (przeciągnięcie) | Przesuwanie obrazu |
| **PPM** (przeciągnięcie) | Rysowanie nowej ramki |
| **Scroll** | Zoom in/out w punkcie kursora |

### Klawiatura — Nawigacja Ogólna

| Klawisz | Akcja |
|---|---|
| `Enter` (bez zaznaczenia) | Zapisz i przejdź do następnego zdjęcia |
| `L` | Otwórz listę plików |
| `A` | Zatwierdź wszystkie podpowiedzi YOLO |
| `R` | Odrzuć wszystkie podpowiedzi YOLO |
| `1` | Oznacz wszystkie ramki jako OK |
| `2` | Oznacz wszystkie ramki jako Odrzucone |
| `F` | Dopasuj widok do okna |
| `0` | Skala 1:1 (100%) |
| `Esc` / `Q` | Zamknij bez zapisu |

### Klawiatura — Edycja Zaznaczonej Ramki

| Klawisz | Akcja |
|---|---|
| `Enter` | Zatwierdź ramkę i przejdź do następnej |
| `Del` | Usuń zaznaczoną ramkę |
| `Backspace` | Usuń ostatni znak etykiety (lub usuń ramkę gdy pusta) |
| `Ctrl+V` | Wklej etykietę ze schowka (polskie znaki) |
| Pisanie | Wpisuje etykietę tekstową do ramki |

### Nawigacja w Liście Plików (`L`)

| Klawisz | Akcja |
|---|---|
| `↑` / `↓` | O 1 pozycję |
| `Page Up` / `Page Down` | O 10 pozycji |
| `Home` / `End` | Pierwszy / ostatni plik |
| Cyfry + `Enter` | Skok bezpośrednio do numeru pliku |
| `Esc` / `L` | Zamknij listę |

### Kolory Ramek

| Kolor | Znaczenie |
|---|---|
| 🟢 Zielony | Zatwierdzona ramka |
| 🟡 Żółty przerywany | Podpowiedź YOLO (niezatwierdzona) |
| 🟠 Pomarańczowy | Ramka dodana ręcznie |
| ⬜ Szary | Ramka odrzucona |
| 🔵 Cyjanowy (jasny) | Ramka aktualnie zaznaczona |

---

## 4. Struktura Wyników

Po wykonaniu `--extract` wszystkie pliki trafiają płasko do folderu `output/`:

```
output/
  doc001_f001.jpg
  doc001_f002.jpg
  doc002_f001.jpg
  ...
  summary.csv
```

Format `summary.csv` (UTF-8, bez nagłówka):

```
doc001_f001.jpg,Kowalski
doc001_f002.jpg,Józefa
doc002_f001.jpg,Nowak
```

Po normalizacji wycinki trafiają do:

```
outputNormalized/              ← wycinki przeskalowane do jednolitego rozmiaru
OutputNormalized_Without_Reshape/  ← wycinki w oryginalnym rozmiarze
```

---

## 5. Zaawansowana Konfiguracja

### Parametry detekcji HSV (`red_box_tool/core/config.py`)

| Parametr | Opis |
|---|---|
| `HSV_LOWER1/2`, `HSV_UPPER1/2` | Zakres koloru czerwonego w HSV |
| `MIN_BOX_W/H` | Minimalny rozmiar wykrywanych ramek |
| `SAVE_DEBUG_IMAGES` | `False` = wyłącz maski diagnostyczne (szybsze działanie) |

### Parametry YOLO (`red_box_tool/detection/yolo_trainer.py`)

| Parametr | Opis |
|---|---|
| `MIN_IMAGES_TO_TRAIN` | Minimalna liczba próbek do pierwszego treningu (domyślnie: 3) |
| `CONFIDENCE_THRESHOLD` | Minimalny próg pewności podpowiedzi (domyślnie: 0.35) |
| `BASE_MODEL` | Bazowy model YOLO (domyślnie: `yolov8n.pt`) |
