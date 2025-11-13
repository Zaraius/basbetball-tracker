from ultralytics import YOLO
import cv2
import numpy as np
import os
import re

class BasketballSegmentation:
    def __init__(self, model_path, input_folder, output_folder):
        self.model = YOLO(model_path)
        self.input_folder = input_folder
        self.output_folder = output_folder
        os.makedirs(self.output_folder, exist_ok=True)

    def process_folder(self):
        image_files = [
            f for f in os.listdir(self.input_folder)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]

        # Sort numerically if filenames contain numbers
        def numerical_key(f):
            numbers = re.findall(r'\d+', f)
            return int(numbers[0]) if numbers else float('inf')
        
        image_files.sort(key=numerical_key)

        all_radii = []

        for filename in image_files:
            image_path = os.path.join(self.input_folder, filename)
            print(f"\nProcessing {image_path}")
            radii = self.process_image(image_path)
            if radii:
                all_radii.extend(radii)
        
        return all_radii

    def process_image(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error loading image: {image_path}")
            return []

        results = self.model(image)
        result = results[0]
        boxes = result.boxes

        if len(boxes) == 0:
            print(f"No detections in {image_path}")
            return []

        output = image.copy()
        radii = []

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            pad = 5
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(image.shape[1], x2 + pad)
            y2 = min(image.shape[0], y2 + pad)
            cropped = image[y1:y2, x1:x2]

            if cropped.size == 0:
                continue

            gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)

            circles = cv2.HoughCircles(
                gray,
                cv2.HOUGH_GRADIENT,
                dp=1.5,
                minDist=1000,
                param1=100,
                param2=50,
                minRadius=20,
                maxRadius=0
            )

            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)

            if circles is not None:
                circles = np.uint16(np.around(circles))
                for circle in circles[0, :]:
                    cx, cy, radius = circle
                    cx_full, cy_full = cx + x1, cy + y1
                    radii.append(radius)
                    cv2.circle(output, (cx_full, cy_full), radius, (255, 0, 0), 3)
                    cv2.circle(output, (cx_full, cy_full), 2, (0, 0, 255), 3)
                    area = np.pi * (radius ** 2)
                    print(f"{os.path.basename(image_path)} | radius={radius}px, area={area:.2f}px²")

        output_name = os.path.splitext(os.path.basename(image_path))[0] + '_processed.jpg'
        output_path = os.path.join(self.output_folder, output_name)
        cv2.imwrite(output_path, output)
        print(f"Saved: {output_path}")

        return radii


# Example usage
if __name__ == "__main__":
    segmentation = BasketballSegmentation(
        model_path="best2.pt",
        input_folder="distance_calibration_photos",
        output_folder="distance_output"
    )

    all_radii = segmentation.process_folder()
    print("\nAll detected radii:", all_radii)
