"""
download_models.py
==================
Downloads trained .keras model files from Google Drive at startup.
This is needed because Render's free tier cannot train models —
we upload the trained files to Google Drive and download them here.

Steps to get your Google Drive file IDs:
  1. Upload your .keras file to Google Drive
  2. Right-click → Share → Change to "Anyone with the link"
  3. Copy the link — it looks like:
     https://drive.google.com/file/d/1ABC123XYZ.../view
  4. The file ID is the part between /d/ and /view → "1ABC123XYZ..."
  5. Paste that ID below
"""

import os
import logging

log = logging.getLogger(__name__)

# ── Paste your Google Drive file IDs here ────────────────────────────────────
PNEUMONIA_FILE_ID = "1KEZ0MVCTdCbKGVxp9FEPe5-nYYMeHhvZ"
SKIN_FILE_ID      = "1IXSt7wT8tfLSMr05Sdhx9q5HcV89ru6v"

MODEL_FILES = {
    "pneumonia_mobilenet.keras":      PNEUMONIA_FILE_ID,
    "skin_mobilenet.keras":           SKIN_FILE_ID,
}

# skin_mobilenet_classes.json is small — commit it directly to GitHub
# (remove it from .gitignore first)


def download_models():
    """Download model files from Google Drive if not already present."""
    try:
        import gdown
    except ImportError:
        log.error("gdown not installed. Run: pip install gdown")
        return

    for filename, file_id in MODEL_FILES.items():
        if os.path.exists(filename):
            log.info("✓ %s already exists — skipping download.", filename)
            continue

        if "PASTE_" in file_id:
            log.warning("File ID not set for %s — skipping.", filename)
            continue

        url = f"https://drive.google.com/uc?id={file_id}"
        log.info("Downloading %s from Google Drive…", filename)
        try:
            gdown.download(url, filename, quiet=False)
            log.info("✓ %s downloaded successfully.", filename)
        except Exception as exc:
            log.error("Failed to download %s: %s", filename, exc)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    download_models()