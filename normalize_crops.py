"""
normalize_crops.py
------------------
Na podstawie pliku extremes.json przeskalowuje wszystkie wycinki
do maksymalnej szerokości i maksymalnej wysokości (bez zachowania proporcji).
Wszystkie wycinki trafiają do katalogu outputNormalized/ (bez podfolderów).
Kopiuje summary.csv zachowując format: file_name, label (bez nagłówka, utf-8).
"""

import csv
import json
import shutil
from pathlib import Path

import cv2

OUTPUT_DIR = Path("output")
OUTPUT_NORMALIZED_DIR = Path("outputNormalized")
EXTREMES_FILE = Path("extremes.json")


def copy_summary(output_dir: Path, result_dir: Path):
    src = output_dir / "summary.csv"
    dst = result_dir / "summary.csv"

    if not src.exists():
        print("  UWAGA: Brak summary.csv w folderze output - pomijam.")
        return

    shutil.copy2(str(src), str(dst))
    print(f"  Skopiowano summary.csv -> {dst}")


def normalize_crops(output_dir: Path, result_dir: Path, extremes_file: Path):
    if not extremes_file.exists():
        print(f"BŁĄD: Nie znaleziono {extremes_file}. Uruchom najpierw find_extremes.py")
        return

    with open(extremes_file, "r", encoding="utf-8") as f:
        extremes = json.load(f)

    target_w = extremes["najszerszy"]["szerokosc_px"]
    target_h = extremes["najwyzszy"]["wysokosc_px"]

    print(f"Rozmiar docelowy: {target_w} x {target_h} px")

    # Pliki płasko w output/ (eksporter nie tworzy podfolderów)
    all_images = list(output_dir.glob("*.jpg"))
    if not all_images:
        print(f"Brak plików JPG w {output_dir}")
        return

    print(f"Znaleziono {len(all_images)} wycinków. Normalizuję...")

    result_dir.mkdir(parents=True, exist_ok=True)
    processed = 0
    skipped = 0

    for img_path in all_images:
        img = cv2.imread(str(img_path))
        if img is None:
            skipped += 1
            continue

        dst_path = result_dir / img_path.name

        resized = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
        cv2.imwrite(str(dst_path), resized)
        processed += 1

        if processed % 100 == 0:
            print(f"  Przetworzono {processed}/{len(all_images)}...")

    copy_summary(output_dir, result_dir)

    print(f"\n=== GOTOWE ===")
    print(f"  Przetworzono: {processed} wycinków")
    print(f"  Pominięto:    {skipped} (błąd odczytu)")
    print(f"  Lokalizacja:  {result_dir.resolve()}")
    print(f"  Rozmiar:      {target_w} x {target_h} px")


if __name__ == "__main__":
    normalize_crops(OUTPUT_DIR, OUTPUT_NORMALIZED_DIR, EXTREMES_FILE)