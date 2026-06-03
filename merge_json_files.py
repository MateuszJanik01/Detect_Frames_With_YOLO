import json
import os
from collections import defaultdict

# Folder zawierający pliki approved.json do scalenia
INPUT_DIR = "../dane_do_scalenia"
# Wynikowy plik
OUTPUT_FILE = "approved_merged.json"

# Czy usuwać duplikaty ramek (True = bezpieczniej, False = szybciej)
DEDUPLICATE = False

merged = defaultdict(list)
total_files = 0
total_frames = 0

for root, _, files in os.walk(INPUT_DIR):
    for file in files:
        if file == "approved.json":
            path = os.path.join(root, file)
            print(f"Wczytywanie: {path}")

            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except json.JSONDecodeError as e:
                print(f"  BŁĄD parsowania JSON: {e} — pominięto plik")
                continue
            except Exception as e:
                print(f"  BŁĄD odczytu: {e} — pominięto plik")
                continue

            if not isinstance(data, dict):
                print(f"  OSTRZEŻENIE: nieoczekiwany format (nie dict) — pominięto plik")
                continue

            file_frames = 0
            for image_name, frames in data.items():
                if not isinstance(frames, list):
                    print(f"  OSTRZEŻENIE: wartość dla '{image_name}' nie jest listą — pominięto klucz")
                    continue
                merged[image_name].extend(frames)
                file_frames += len(frames)

            print(f"  → {len(data)} obrazów, {file_frames} ramek")
            total_files += 1
            total_frames += file_frames

print(f"\nWczytano łącznie: {total_files} plików, {total_frames} ramek")

# Deduplikacja — usuwa identyczne ramki dla tego samego obrazu
if DEDUPLICATE:
    before = sum(len(v) for v in merged.values())
    merged = {
        k: [dict(t) for t in {tuple(sorted(d.items())) for d in v}]
        for k, v in merged.items()
    }
    after = sum(len(v) for v in merged.values())
    print(f"Deduplikacja: {before} → {after} ramek (usunięto {before - after} duplikatów)")
else:
    merged = dict(merged)

# Zapis wyniku
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(merged, f, ensure_ascii=False, indent=2)

print(f"Scalono dane do pliku: {OUTPUT_FILE}")