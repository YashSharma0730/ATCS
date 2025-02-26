from ultralytics import YOLO
import cv2
import os

def detect_number_plate(cropped_vehicle, plate_model, frame_count, i, number_plate_dir):
    """Detect and save the number plate from the cropped vehicle image."""
    plate_results = plate_model(cropped_vehicle)
    for plate_result in plate_results:
        for j, box in enumerate(plate_result.boxes.xyxy):
            x1_plate, y1_plate, x2_plate, y2_plate = map(int, box)
            number_plate = cropped_vehicle[y1_plate:y2_plate, x1_plate:x2_plate]
            number_plate_image_path = os.path.join(number_plate_dir, f"plate_{frame_count}_{i}.jpg")
            cv2.imwrite(number_plate_image_path, number_plate)
            print(f"Saved number plate image to {number_plate_image_path}")
            return number_plate_image_path
    return None
