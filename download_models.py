"""
download_models.py
==================
Downloads trained .keras model files from Google Drive at startup.
"""

import os
import logging

log = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PNEUMONIA_FILE_ID = "1KEZ0MVCTdCbKGVxp9FEPe5-nYYMeHhvZ"
SKIN_FILE_ID      = "1IXSt7wT8tfLSMr05Sdhx9q5HcV89ru6v"

MODEL_FILES = {
    os.path.join(BASE_DIR, "pneumonia_mobilenet.keras"): PNEUMONIA_FILE_ID,
    os.path.join(BASE_DIR, "skin_mobilenet.keras"):      SKIN_FILE_ID,
}


def download_models():
    """Download model files from Google Drive if not already present."""
    try:
        import gdown
    except ImportError:
        log.error("gdown not installed. Run: pip install gdown")
        return

    for filepath, file_id in MODEL_FILES.items():
        if os.path.exists(filepath):
            log.info("✓ %s already exists — skipping download.", filepath)
            continue

        url = f"https://drive.google.com/uc?id={file_id}"
        log.info("Downloading %s from Google Drive…", os.path.basename(filepath))
        try:
            gdown.download(url, filepath, quiet=False)
            log.info("✓ %s downloaded successfully.", os.path.basename(filepath))
        except Exception as exc:
            log.error("Failed to download %s: %s", os.path.basename(filepath), exc)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    download_models()