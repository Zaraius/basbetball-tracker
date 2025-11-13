import cv2
import numpy as np
import os
import re

class BasketballSegmentation:
    def __init__(self, input_folder, output_folder):
        self.input_images_folder = input_folder + "/images"
        self.input_labels_folder = input_folder + "/labels"
        self.output_folder = output_folder
        os.makedirs(self.output_folder, exist_ok=True)

    def process_folder(self):
        image_files = [
            f for f in os.listdir(self.input_images_folder)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]   

        # Sort numerically (frame_00001.jpg → 1, frame_00002.jpg → 2, etc.)
        def numerical_key(f):
            numbers = re.findall(r'\d+', f)
            return int(numbers[0]) if numbers else float('inf')
        
        image_files.sort(key=numerical_key)

        all_radii = []
        prev_radius = None
        for filename in image_files:
            image_path = os.path.join(self.input_images_folder, filename)
            label_path = os.path.join(self.input_labels_folder, os.path.splitext(filename)[0] + ".txt")
            prev_radius = self.process_image(image_path, label_path, prev_radius)
            all_radii.append(prev_radius)
        return all_radii

    def process_image(self, image_path, label_path, prev_radius=None):
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error loading image: {image_path}")
            return prev_radius if prev_radius is not None else 0

        h, w, _ = image.shape
        output = image.copy()
        radius_to_use = None

        # Read YOLO labels
        with open(label_path, "r") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        if lines:
            # Pick first detection
            cls, x_center, y_center, bbox_w, bbox_h = map(float, lines[0].split()[:5])
            x_center_px = int(x_center * w)
            y_center_px = int(y_center * h)
            bbox_w_px = int(bbox_w * w)
            bbox_h_px = int(bbox_h * h)

            x1 = max(0, x_center_px - bbox_w_px // 2)
            y1 = max(0, y_center_px - bbox_h_px // 2)
            x2 = min(w, x_center_px + bbox_w_px // 2)
            y2 = min(h, y_center_px + bbox_h_px // 2)

            cropped = image[y1:y2, x1:x2]
            gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)

            circles = cv2.HoughCircles(
                gray, cv2.HOUGH_GRADIENT, dp=1.5, minDist=1000,
                param1=100, param2=50, minRadius=20, maxRadius=0
            )

            # Prefer circle detection
            if circles is not None:
                circles = np.uint16(np.around(circles))
                radius_to_use = int(circles[0,0,2])
            else:
                # fallback to bounding box radius
                radius_to_use = int(max(bbox_w_px, bbox_h_px) / 2)

            # draw box & circle for visualization
            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
            if radius_to_use:
                cv2.circle(output, (x_center_px, y_center_px), radius_to_use, (255,0,0), 2)

        else:
            # no label, keep previous or 0
            radius_to_use = prev_radius if prev_radius is not None else 0

        # Save visualization
        output_name = os.path.splitext(os.path.basename(image_path))[0] + '_processed.jpg'
        output_path = os.path.join(self.output_folder, output_name)
        cv2.imwrite(output_path, output)
        print(f"Saved: {output_path} | radius={radius_to_use}")
        print(radius_to_use)
        return radius_to_use
