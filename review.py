"""
review.py
---------
Interaktywne narzędzie do weryfikacji wykrytych ramek oraz finalnej ekstrakcji danych.
Rozszerzone o integrację z YOLOv8: predykcje jako podpowiedzi + przyrostowy trening.
"""

import argparse
from pathlib import Path

import cv2

from red_box_tool.core.config import APPROVED_FILE as DEFAULT_APPROVED_FILE
from red_box_tool.core.config import CORRECT_COUNTS_FILE, DEBUG_DIR, IMG_DIR
from red_box_tool.core.io_utils import (
    load_approved,
    load_expected_counts,
    save_approved,
)
from red_box_tool.detection.engine import detect_super_hybrid
from red_box_tool.interface.exporter import extract_approved
from red_box_tool.interface.reviewer import Reviewer

# Import modułu YOLO – opcjonalny (graceful degradation)
try:
    from red_box_tool.detection import yolo_trainer
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

OUTPUT_DIR = Path("output")


def _merge_predictions(existing: list[dict], yolo_preds: list[dict]) -> list[dict]:
    """
    Łączy istniejące ramki (z detekcji HSV lub poprzedniej sesji)
    z predykcjami YOLO, unikając duplikatów (IoU > 25%).

    Predykcje YOLO są dołączane NA KOŃCU listy, oznaczone etykietą 'yolo:XX'.
    Użytkownik widzi je jako sugestie i może je zatwierdzić lub usunąć.
    """
    from red_box_tool.core.geometry import iou

    merged = list(existing)
    for pred in yolo_preds:
        pred_rect = (pred["x"], pred["y"], pred["w"], pred["h"])
        # Sprawdź nakładanie z już istniejącymi ramkami
        overlap = any(
            iou(pred_rect, (e["x"], e["y"], e["w"], e["h"])) > 0.25
            for e in merged
        )
        if not overlap:
            merged.append(pred)
    return merged


def main() -> None:
    """Główna funkcja narzędzia do przeglądu i eksportu."""
    parser = argparse.ArgumentParser(
        description="Narzędzie do przeglądu i etykietowania ramek."
    )
    parser.add_argument(
        "--image", "-i", default=None, help="Ścieżka do konkretnego obrazu."
    )
    parser.add_argument(
        "--extract",
        "-e",
        nargs="?",
        const=str(DEFAULT_APPROVED_FILE),
        default=None,
        help="Wyodrębnij ramki z pliku JSON.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Usuń plik approved.json i zacznij od nowa.",
    )
    parser.add_argument(
        "--no-yolo",
        action="store_true",
        help="Wyłącz podpowiedzi i trening YOLO.",
    )
    args = parser.parse_args()

    use_yolo = YOLO_AVAILABLE and not args.no_yolo

    if use_yolo:
        print(f"  [YOLO] {yolo_trainer.get_status()}")
    else:
        if not YOLO_AVAILABLE:
            print("  [YOLO] Moduł niedostępny - tryb klasyczny.")

    if args.reset and DEFAULT_APPROVED_FILE.exists():
        DEFAULT_APPROVED_FILE.unlink()
        print(f"Usunięto {DEFAULT_APPROVED_FILE}")
        return

    # Automatyczne uruchomienie detekcji, jeśli brakuje pliku wynikowego
    if not DEFAULT_APPROVED_FILE.exists():
        if args.extract is None or Path(args.extract) == DEFAULT_APPROVED_FILE:
            print(
                f"\n[INFO] Plik {DEFAULT_APPROVED_FILE} nie istnieje. Uruchamiam detekcję automatyczną..."
            )
            import detect
            detect.main()

    # Tryb EKSTRAKCJI
    if args.extract is not None:
        extract_file = Path(args.extract)
        if not extract_file.exists():
            print(f"BŁĄD: Plik {extract_file} nie istnieje.")
            return
        data = load_approved(extract_file)
        if data:
            extract_approved(data, IMG_DIR, OUTPUT_DIR, DEBUG_DIR, approved_path=extract_file)
        return

    # Tryb PRZEGLĄDU
    approved_data = load_approved(DEFAULT_APPROVED_FILE)
    expected_counts = load_expected_counts(CORRECT_COUNTS_FILE)

    if args.image:
        images = [Path(args.image)]
    else:
        extensions = ["jpg", "JPG", "jpeg", "png", "bmp"]
        images = sorted({p for ext in extensions for p in IMG_DIR.glob(f"*.{ext}")})

    if not images:
        print(f"Brak obrazów do przetworzenia w {IMG_DIR}/")
        return

    print(f"Znaleziono {len(images)} obrazów. Rozpoczynanie przeglądu...")
    all_fnames = [p.name for p in images]

    cur_idx = 0
    total = len(images)

    while 0 <= cur_idx < total:
        img_path = images[cur_idx]
        fname = img_path.name
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"  [BŁĄD] Nie można otworzyć: {fname}")
            cur_idx += 1
            continue

        existing = approved_data.get(fname, [])
        target = expected_counts.get(fname)

        # Detekcja HSV jeśli brak danych
        if not existing and fname in expected_counts:
            print(f"\n[{fname}] Detekcja automatyczna HSV (cel: {target})...")
            existing = detect_super_hybrid(
                image, target, debug_name=img_path.stem, debug_dir=DEBUG_DIR
            )
        else:
            print(
                f"\n[{fname}] ({cur_idx+1}/{total}) Wczytywanie ramek ({len(existing)} wpisów)"
            )

        # ----------------------------------------------------------------
        # YOLO: dołącz predykcje jako podpowiedzi (jeśli model jest gotowy)
        # ----------------------------------------------------------------
        if use_yolo and yolo_trainer.is_model_ready():
            yolo_preds = yolo_trainer.predict(image)
            if yolo_preds:
                print(f"  [YOLO] Dodaję {len(yolo_preds)} podpowiedzi (żółte ramki)")
                existing = _merge_predictions(existing, yolo_preds)
            else:
                print("  [YOLO] Brak pewnych predykcji dla tego obrazu.")
        elif use_yolo:
            n = yolo_trainer.count_training_samples()
            from red_box_tool.detection.yolo_trainer import MIN_IMAGES_TO_TRAIN
            print(f"  [YOLO] Model jeszcze nie gotowy ({n}/{MIN_IMAGES_TO_TRAIN} próbek)")

        # Uruchomienie GUI
        reviewer = Reviewer(image, fname, existing, target=target, all_files=all_fnames)
        was_saved = reviewer.run()

        # Zapisanie wyników
        result = reviewer.get_result()
        approved_data[fname] = result
        save_approved(approved_data, DEFAULT_APPROVED_FILE)

        # ----------------------------------------------------------------
        # YOLO: zapisz próbkę treningową i uruchom fine-tuning w tle
        # ----------------------------------------------------------------
        if use_yolo and was_saved:
            # Filtruj tylko zatwierdzone ramki (bez predykcji YOLO które nie zostały potwierdzone)
            confirmed = [
                b for b in result
                if b.get("ok", True) and not b.get("deleted", False)
                # Nie trenuj na niepoprawionych predykcjach YOLO
                # (label zaczyna się od 'yolo:' = niezweryfikowane)
                and not b.get("label", "").startswith("yolo:")
            ]
            if confirmed:
                saved = yolo_trainer.save_sample(fname, image, confirmed)
                if saved:
                    n = yolo_trainer.count_training_samples()
                    print(f"  [YOLO] Próbka zapisana ({n} łącznie). Uruchamiam trening w tle...")
                    yolo_trainer.trigger_training_async(
                        on_done=lambda: print(f"  [YOLO] {yolo_trainer.get_status()}")
                    )

        # Nawigacja
        if reviewer.requested_file:
            try:
                cur_idx = all_fnames.index(reviewer.requested_file)
                print(f"  -> Skok do: {reviewer.requested_file}")
            except ValueError:
                cur_idx += 1
        elif not was_saved:
            print("  Wychodzenie.")
            break
        else:
            cur_idx += 1

    print("\nPrzegląd ukończony.")


if __name__ == "__main__":
    main()