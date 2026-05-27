"""
exporter.py
-----------
Moduł odpowiedzialny za wycinanie fragmentów obrazów i generowanie raportów CSV.
Zoptymalizowany pod kątem szybkości poprzez przetwarzanie równoległe.
Wszystkie wycinki zapisywane są płasko w jednym folderze (bez podfolderów).
Nazewnictwo: nazwapliku_f001.jpg
Przed eksportem usuwa z approved.json obiekty z pustą etykietą.
"""

import csv
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Optional

import cv2


def clean_empty_labels(data: dict, approved_path: Path) -> dict:
    """
    Usuwa z danych wszystkie ramki z pustą etykietą (label: ""). oraz z etykietą zaczynającą się od "yolo:".
    Zapisuje oczyszczony plik approved.json.
    Zwraca oczyszczony słownik.
    """
    total_before = sum(len(boxes) for boxes in data.values())
    removed = 0

    cleaned = {}
    for fname, boxes in data.items():
        kept = [b for b in boxes if (b.get("label", "").strip() and not b.get("label", "").strip().lower().startswith("yolo:"))]
        removed += len(boxes) - len(kept)
        if kept:
            cleaned[fname] = kept

    total_after = sum(len(boxes) for boxes in cleaned.values())

    with open(approved_path, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, indent=2, ensure_ascii=False)

    print(f"  [Czyszczenie] Usunięto {removed} ramek z pustą etykietą")
    print(f"  [Czyszczenie] Pozostało {total_after} z {total_before} ramek")
    print(f"  [Czyszczenie] Zapisano: {approved_path.name}")

    return cleaned


def _process_image_crops(args) -> list:
    """
    Pomocnicza funkcja do przetwarzania ramek dla pojedynczego obrazu.
    Zapisuje wycinki bezpośrednio w output_dir (bez podfolderów).
    Nazewnictwo: nazwapliku_f001.jpg
    """
    fname, boxes, img_dir, output_dir = args
    img_path = img_dir / fname
    img = cv2.imread(str(img_path))
    if img is None:
        return []

    metadata = []
    img_h, img_w = img.shape[:2]
    frame_idx = 1
    stem = img_path.stem  # nazwa pliku bez rozszerzenia

    for b in boxes:
        if not b.get("ok", True) or b.get("deleted", False):
            continue

        x = max(0, int(b["x"]))
        y = max(0, int(b["y"]))
        w = min(int(b["w"]), img_w - x)
        h = min(int(b["h"]), img_h - y)

        if w <= 0 or h <= 0:
            continue

        crop = img[y:y + h, x:x + w]
        if crop.size == 0:
            continue

        frame_name = f"f{frame_idx:03d}"
        # Nazwa pliku: doc001_f001.jpg
        file_name = f"{stem}_{frame_name}.jpg"
        label = b.get("label", "")

        cv2.imwrite(str(output_dir / file_name), crop)

        metadata.append({
            "file_name": file_name,
            "label": label,
        })
        frame_idx += 1

    return metadata


def extract_approved(data: dict, img_dir: Path, output_dir: Path, debug_dir: Path,
                     approved_path: Optional[Path] = None):
    """
    Główna funkcja eksportu.
    1. Usuwa ramki z pustą etykietą z approved.json i zapisuje plik.
    2. Eksportuje wycinki płasko do output_dir i tworzy summary.csv.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Krok 1: Wyczyść puste etykiety
    if approved_path and approved_path.exists():
        print("\n[Krok 1/2] Czyszczenie pustych etykiet...")
        data = clean_empty_labels(data, approved_path)
    else:
        print("\n[Krok 1/2] Brak ścieżki do approved.json - pomijam czyszczenie.")

    # Krok 2: Eksport
    print(f"\n[Krok 2/2] Eksport dla {len(data)} plików...")

    tasks = [(fname, boxes, img_dir, output_dir) for fname, boxes in data.items()]

    all_metadata = []
    with ProcessPoolExecutor() as executor:
        per_image_results = list(executor.map(_process_image_crops, tasks))

    for results in per_image_results:
        all_metadata.extend(results)

    if all_metadata:
        summary_path = output_dir / "summary.csv"
        with open(summary_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["file_name", "label"])
            writer.writerows(all_metadata)

    print(f"\nUkończono ekstrakcję.")
    print(f"  - Lokalizacja: {output_dir}")
    print(f"  - Liczba ramek: {len(all_metadata)}")
    print(f"  - Raport zbiorczy: {output_dir / 'summary.csv'}")