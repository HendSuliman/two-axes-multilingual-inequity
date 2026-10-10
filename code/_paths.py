"""Repository paths, so every script runs from any working directory.

Run scripts as `python code/<script>.py` from the repository root (or from anywhere).
Inputs are read from data/ and results/; anything a script regenerates is written to
outputs/, so the shipped results/ are never overwritten and can be diffed against.
"""
import zipfile
from pathlib import Path

REPO     = Path(__file__).resolve().parent.parent
DATA     = REPO / "data"
METADATA = DATA / "metadata"
RESULTS  = REPO / "results"
OUTPUTS  = REPO / "outputs"
FLORES   = REPO / "flores200_dataset" / "devtest"

# Corrected token counts (mT5 / XLM-R / Aya-101 recounted with the reference sentencepiece
# library; see README "Changes since the August submission"). The v2 file is kept for audit.
FERTILITY = RESULTS / "fertility_production_v3_spm.csv"


def corpus_dir() -> Path:
    """Unzip data/translations.zip into ./corpus at the repo root on first use."""
    out = REPO / "corpus"
    if not any(out.glob("*.csv")):
        out.mkdir(exist_ok=True)
        with zipfile.ZipFile(DATA / "translations.zip") as z:
            z.extractall(out)
        inner = [p for p in out.iterdir() if p.is_dir() and any(p.glob("*.csv"))]
        if inner:                      # archive with a top-level folder
            return inner[0]
    return out


def outputs() -> Path:
    OUTPUTS.mkdir(exist_ok=True)
    return OUTPUTS


def flores_dir() -> Path:
    """Download and unpack FLORES-200 (CC-BY-SA 4.0) from Meta's official URL on first use."""
    if not FLORES.exists():
        import tarfile, urllib.request
        tgz = REPO / "flores200_dataset.tar.gz"
        urllib.request.urlretrieve("https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz", tgz)
        with tarfile.open(tgz) as t:
            t.extractall(REPO)
    return FLORES
