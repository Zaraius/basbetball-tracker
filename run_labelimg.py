import os
import subprocess

def run_labelimg():
    # Set the paths for the image and annotation directories
    image_dir = "output"
    annotation_classes = "classes.txt"

    # Run labelImg
    subprocess.run(["labelImg", image_dir, annotation_classes, image_dir])

if __name__ == "__main__":
    run_labelimg()
