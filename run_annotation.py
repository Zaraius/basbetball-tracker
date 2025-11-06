import os
from ultralytics import YOLO
from pathlib import Path

# ───────────── CONFIG ───────────── #
IMAGES_FOLDER = 'distance_calibration_photos'          # folder with images
MODEL_FILE    = 'yolo11n.pt'
BATCH_SIZE    = 16                       # adjust to GPU RAM
# ─────────────────────────────────── #

model = YOLO(MODEL_FILE)                 # load once

def annotate_images():
    """
    Runs YOLO in streamed, batched mode.
    - `batch` keeps the GPU busy.
    - `stream=True` yields results immediately so they’re
      not all stored in memory at once.
    """
    for _ in model.predict(
            source=IMAGES_FOLDER,
            batch=BATCH_SIZE,
            stream=True,     # generator – minimises RAM
            save_txt=True,   # one .txt per image
            save=True,      # don’t duplicate images
            verbose=False):  # progress bar off (optional)
        pass                 # nothing else to do; files are saved

    print("✓ Finished annotating folder.")

if __name__ == "__main__":
    annotate_images()
