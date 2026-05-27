"""
copy_crops_original.py
----------------------
Kopiuje wszystkie wycinki z folderu output/ do OutputNormalized_Without_Reshape/
bez zmiany rozmiaru - zachowuje oryginalny rozmiar każdego wycinka.
Kopiuje summary.csv z kolumnami: path, label.
"""

import csv
import shutil
from pathlib import Path

OUTPUT_DIR = Path("output")
OUTPUT_ORIGINAL_DIR = Path("OutputNormalized_Without_Reshape")


def copy_summary_with_path(output_dir: Path, result_dir: Path):
    src = output_dir / "summary.csv"
    dst = result_dir / "summary.csv"

    if not src.exists():
        print("  UWAGA: Brak summary.csv w folderze output - pomijam.")
        return

    with open(src, "r", encoding="utf-8-sig", newline="") as f_in, \
         open(dst, "w", encoding="utf-8-sig", newline="") as f_out:

        reader = csv.DictReader(f_in)
        writer = csv.writer(f_out)

        for row in reader:
            original_stem = Path(row["original_file"]).stem
            filename = f"{original_stem}_{row['frame_name']}.jpg"
            writer.writerow([filename, row["label"]])

    print(f"  Skopiowano summary.csv (path, label) -> {dst}")


def copy_crops(output_dir: Path, result_dir: Path):
    all_images = list(output_dir.rglob("*.jpg"))
    if not all_images:
        print(f"Brak plików JPG w {output_dir}")
        return

    print(f"Znaleziono {len(all_images)} wycinków. Kopiuję...")

    result_dir.mkdir(parents=True, exist_ok=True)
    processed = 0
    skipped = 0

    for img_path in all_images:
        if not img_path.exists():
            skipped += 1
            continue

        # Nowa nazwa: nazwa_podfolderu_nazwa_pliku.jpg
        parent_name = img_path.parent.name
        new_filename = f"{parent_name}_{img_path.name}"
        dst_path = result_dir / new_filename

        shutil.copy2(str(img_path), str(dst_path))
        processed += 1

        if processed % 100 == 0:
            print(f"  Skopiowano {processed}/{len(all_images)}...")

    copy_summary_with_path(output_dir, result_dir)

    print(f"\n=== GOTOWE ===")
    print(f"  Skopiowano:  {processed} wycinków (oryginalny rozmiar)")
    print(f"  Pominięto:   {skipped} (błąd odczytu)")
    print(f"  Lokalizacja: {result_dir.resolve()}")


if __name__ == "__main__":
    copy_crops(OUTPUT_DIR, OUTPUT_ORIGINAL_DIR)