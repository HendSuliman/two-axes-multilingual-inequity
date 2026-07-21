#!/usr/bin/env python3
"""Fetch the Tanzil Uthmani Arabic source text (not bundled; see DATA_LICENSE.md).

Downloads the verse-per-line Uthmani text from tanzil.net and writes
data/arabic_source_uthmani.csv with columns sura, aya, text (6,236 rows,
Hafs verse count, matching the corpus alignment keys).

Tanzil terms: use and redistribution of the unmodified text with attribution
(https://tanzil.net). Keep the text unmodified.
"""
import csv, urllib.request, urllib.parse
from pathlib import Path

URL = ("https://tanzil.net/pub/download/index.php?" +
       urllib.parse.urlencode({"quranType": "uthmani", "outType": "txt", "agree": "true"}))

def main():
    raw = urllib.request.urlopen(URL, timeout=60).read().decode("utf-8")
    out = Path(__file__).resolve().parent.parent / "data" / "arabic_source_uthmani.csv"
    n = 0
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["sura", "aya", "text"])
        for line in raw.splitlines():
            line = line.strip()
            if not line or line.startswith("#"): continue
            parts = line.split("|")
            if len(parts) != 3: continue
            w.writerow(parts); n += 1
    assert n == 6236, f"expected 6236 verses, got {n}"
    print(f"wrote {out} ({n} verses)")

if __name__ == "__main__":
    main()
