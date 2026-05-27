"""
find_extremes.py
----------------
Przechodzi po wszystkich wycinkach w folderze output/
i znajduje najszerszy oraz najwyższy wycinek.
Wynik zapisuje do pliku JSON.
"""

import json
from pathlib import Path

import cv2

OUTPUT_DIR = Path("output")
RESULT_FILE = Path("extremes.json")


def find_extreme_crops(output_dir: Path, result_file: Path):
    all_images = list(output_dir.rglob("*.jpg"))

    if not all_images:
        print(f"Brak plików JPG w {output_dir}")
        return

    print(f"Znaleziono {len(all_images)} wycinków. Analizuję...")

    widest_path = None
    widest_w = 0
    widest_h = 0

    tallest_path = None
    tallest_w = 0
    tallest_h = 0

    for img_path in all_images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        if w > widest_w:
            widest_w = w
            widest_h = h
            widest_path = img_path

        if h > tallest_h:
            tallest_h = h
            tallest_w = w
            tallest_path = img_path

    result = {
        "najszerszy": {
            "plik": str(widest_path),
            "szerokosc_px": widest_w,
            "wysokosc_px": widest_h,
        },
        "najwyzszy": {
            "plik": str(tallest_path),
            "szerokosc_px": tallest_w,
            "wysokosc_px": tallest_h,
        },
        "przeanalizowano_wycinki": len(all_images),
    }

    with open(result_file, "w", encoding="utf-8-sig") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print("\n=== WYNIKI ===")
    print(f"Najszerszy wycinek:")
    print(f"  Plik:      {widest_path}")
    print(f"  Rozmiar:   {widest_w} x {widest_h} px")
    print(f"\nNajwyższy wycinek:")
    print(f"  Plik:      {tallest_path}")
    print(f"  Rozmiar:   {tallest_w} x {tallest_h} px")
    print(f"\nZapisano wyniki do: {result_file}")


if __name__ == "__main__":
    find_extreme_crops(OUTPUT_DIR, RESULT_FILE)