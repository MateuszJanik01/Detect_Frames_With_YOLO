"""
yolo_trainer.py
---------------
Moduł integracji YOLOv8 z narzędziem red_box_tool.
Odpowiada za:
  - konwersję zatwierdzonych ramek do formatu YOLO (txt)
  - przyrostowy fine-tuning modelu po każdym zatwierdzonym obrazie
  - inferencję (predykcje) na nowych obrazach jako podpowiedzi
Trening odbywa się w osobnym wątku, aby nie blokować GUI.
"""

import shutil
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Stałe (nadpisywalne przez config)
# ---------------------------------------------------------------------------
YOLO_WEIGHTS_DIR = Path("yolo_weights")       # katalog z wagami modelu
YOLO_DATASET_DIR = Path("yolo_dataset")       # dane treningowe w formacie YOLO
BASE_MODEL = "yolov8n.pt"                     # bazowy model (nano = szybki na CPU)
TRAINED_MODEL_NAME = "best.pt"               # nazwa zapisywanych wag
MIN_IMAGES_TO_TRAIN = 3                       # ile obrazów potrzeba do 1. treningu
CONFIDENCE_THRESHOLD = 0.35                   # minimalny próg pewności predykcji
FINETUNE_EPOCHS = 5                           # liczba epok przy każdym fine-tuningu
FINETUNE_IMGSZ = 640                          # rozmiar obrazu podczas treningu
CLASS_NAME = "red_box"                        # nazwa klasy (tylko 1 klasa)

# ---------------------------------------------------------------------------
# Ścieżki wewnętrzne
# ---------------------------------------------------------------------------
_IMAGES_DIR = YOLO_DATASET_DIR / "images" / "train"
_LABELS_DIR = YOLO_DATASET_DIR / "labels" / "train"
_DATA_YAML = YOLO_DATASET_DIR / "data.yaml"
_TRAINED_WEIGHTS = YOLO_WEIGHTS_DIR / TRAINED_MODEL_NAME

# ---------------------------------------------------------------------------
# Globalny lock – zapobiega równoległym treningom
# ---------------------------------------------------------------------------
_train_lock = threading.Lock()
_last_train_time: float = 0.0


# ---------------------------------------------------------------------------
# Pomocnicze
# ---------------------------------------------------------------------------

def _ensure_dirs():
    """Tworzy wymagane katalogi jeśli nie istnieją."""
    _IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    _LABELS_DIR.mkdir(parents=True, exist_ok=True)
    YOLO_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)


def _write_data_yaml():
    """Generuje plik data.yaml wymagany przez Ultralytics."""
    content = (
        f"path: {YOLO_DATASET_DIR.resolve()}\n"
        f"train: images/train\n"
        f"val: images/train\n"  # przy małym zbiorze używamy tych samych danych
        f"nc: 1\n"
        f"names: ['{CLASS_NAME}']\n"
    )
    _DATA_YAML.write_text(content, encoding="utf-8")


def _boxes_to_yolo(boxes: list[dict], img_w: int, img_h: int) -> list[str]:
    """
    Konwertuje listę ramek (format x,y,w,h w pikselach) do formatu YOLO:
    <class> <cx_norm> <cy_norm> <w_norm> <h_norm>
    """
    lines = []
    for b in boxes:
        if not b.get("ok", True) or b.get("deleted", False):
            continue
        x, y, w, h = b["x"], b["y"], b["w"], b["h"]
        cx = (x + w / 2) / img_w
        cy = (y + h / 2) / img_h
        wn = w / img_w
        hn = h / img_h
        # Obcinamy do [0,1] na wszelki wypadek
        cx = max(0.0, min(1.0, cx))
        cy = max(0.0, min(1.0, cy))
        wn = max(0.001, min(1.0, wn))
        hn = max(0.001, min(1.0, hn))
        lines.append(f"0 {cx:.6f} {cy:.6f} {wn:.6f} {hn:.6f}")
    return lines


# ---------------------------------------------------------------------------
# Publiczne API
# ---------------------------------------------------------------------------

def save_sample(fname: str, image: np.ndarray, boxes: list[dict]) -> bool:
    """
    Zapisuje jeden przykład treningowy (obraz + etykiety YOLO).
    Zwraca True jeśli zapisano co najmniej jedną ramkę.

    Args:
        fname:  Oryginalna nazwa pliku (np. 'scan001.jpg').
        image:  Macierz obrazu (BGR).
        boxes:  Zatwierdzone ramki w formacie słownikowym.
    """
    _ensure_dirs()
    ih, iw = image.shape[:2]
    lines = _boxes_to_yolo(boxes, iw, ih)
    if not lines:
        return False

    stem = Path(fname).stem
    img_dst = _IMAGES_DIR / f"{stem}.jpg"
    lbl_dst = _LABELS_DIR / f"{stem}.txt"

    cv2.imwrite(str(img_dst), image)
    lbl_dst.write_text("\n".join(lines), encoding="utf-8")
    return True


def count_training_samples() -> int:
    """Zwraca liczbę dostępnych przykładów treningowych."""
    if not _LABELS_DIR.exists():
        return 0
    return sum(1 for f in _LABELS_DIR.glob("*.txt") if f.stat().st_size > 0)


def trigger_training_async(on_done: Optional[callable] = None):
    """
    Uruchamia fine-tuning YOLOv8 w osobnym wątku (nie blokuje GUI).
    Wywołuje on_done() po zakończeniu, jeśli podano.
    Ignoruje wywołanie jeśli trening właśnie trwa.
    """
    if not _train_lock.acquire(blocking=False):
        print("  [YOLO] Trening już trwa - pomijam.")
        return

    def _train():
        global _last_train_time
        try:
            n = count_training_samples()
            if n < MIN_IMAGES_TO_TRAIN:
                print(f"  [YOLO] Za mało próbek ({n}/{MIN_IMAGES_TO_TRAIN}) - pomijam trening.")
                return

            time.sleep(1.0)

            try:
                from ultralytics import YOLO
            except ImportError:
                print("  [YOLO] BŁĄD: Zainstaluj ultralytics: pip install ultralytics")
                return

            _write_data_yaml()

            start_weights = str(_TRAINED_WEIGHTS) if _TRAINED_WEIGHTS.exists() else BASE_MODEL
            print(f"  [YOLO] Rozpoczynam fine-tuning ({n} próbek, {FINETUNE_EPOCHS} epok)...")

            model = YOLO(start_weights)
            model.train(
                data=str(_DATA_YAML),
                epochs=FINETUNE_EPOCHS,
                imgsz=FINETUNE_IMGSZ,
                batch=max(1, min(4, n)),
                device="cpu",
                workers=0,
                project=str(YOLO_WEIGHTS_DIR),
                name="run",
                exist_ok=True,
                verbose=False,
                plots=False,
            )

            # Szukamy best.pt we wszystkich możliwych lokalizacjach
            search_roots = [
                YOLO_WEIGHTS_DIR,
                Path("runs") / "detect" / YOLO_WEIGHTS_DIR.name,
                Path("runs") / "detect",
            ]

            trained_best = None
            for root in search_roots:
                if not root.exists():
                    continue
                candidates = sorted(
                    root.rglob("best.pt"),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
                if candidates:
                    trained_best = candidates[0]
                    break

            if trained_best and trained_best.exists():
                YOLO_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
                # Sprawdź czy źródło i cel to nie ten sam plik
                if trained_best.resolve() != _TRAINED_WEIGHTS.resolve():
                    shutil.copy(str(trained_best), str(_TRAINED_WEIGHTS))
                    print(f"  [YOLO] Wagi skopiowane z: {trained_best}")
                else:
                    print(f"  [YOLO] Wagi już są na miejscu: {_TRAINED_WEIGHTS}")
                print(f"  [YOLO] Trening zakończony pomyślnie! ({n} próbek)")
            else:
                print("  [YOLO] UWAGA: Nie znaleziono wag po treningu.")

            _last_train_time = time.time()

        except Exception as e:
            print(f"  [YOLO] Błąd treningu: {e}")
        finally:
            _train_lock.release()
            if on_done:
                on_done()

    t = threading.Thread(target=_train, daemon=True)
    t.start()


def predict(image: np.ndarray) -> list[dict]:
    """
    Uruchamia inferencję YOLOv8 na obrazie i zwraca listę ramek
    w formacie kompatybilnym z red_box_tool (x, y, w, h, ok, manual, label).

    Zwraca pustą listę jeśli model nie jest jeszcze dostępny.

    Args:
        image: Macierz obrazu BGR.
    """
    if not _TRAINED_WEIGHTS.exists():
        return []

    try:
        from ultralytics import YOLO
    except ImportError:
        return []

    try:
        model = YOLO(str(_TRAINED_WEIGHTS))
        ih, iw = image.shape[:2]

        results = model.predict(
            source=image,
            imgsz=FINETUNE_IMGSZ,
            conf=CONFIDENCE_THRESHOLD,
            device="cpu",
            verbose=False,
        )

        boxes = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                # Współrzędne w pikselach (xyxy)
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w = max(1, x2 - x1)
                h = max(1, y2 - y1)
                conf = float(box.conf[0])
                boxes.append({
                    "x": max(0, x1),
                    "y": max(0, y1),
                    "w": min(w, iw - x1),
                    "h": min(h, ih - y1),
                    "ok": True,
                    "manual": False,
                    "label": f"yolo:{conf:.2f}",
                })

        return boxes

    except Exception as e:
        print(f"  [YOLO] Błąd inferencji: {e}")
        return []


def is_model_ready() -> bool:
    """Sprawdza czy wytrenowany model jest dostępny."""
    return _TRAINED_WEIGHTS.exists()


def get_status() -> str:
    """Zwraca krótki opis stanu modelu (do wyświetlenia w GUI)."""
    n = count_training_samples()
    if not is_model_ready():
        return f"YOLO: brak modelu ({n}/{MIN_IMAGES_TO_TRAIN} próbek)"
    elapsed = time.time() - _last_train_time
    if elapsed < 5:
        return f"YOLO: wytrenowano! ({n} próbek)"
    return f"YOLO: aktywny ({n} próbek)"