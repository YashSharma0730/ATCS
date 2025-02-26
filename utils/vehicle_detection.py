from ultralytics import YOLO
import cv2
import os

def detect_vehicles(frame, model):
    """Detect vehicles in the given frame using YOLOv8."""
    results = model(frame)
    return results

def save_vehicle_image(frame, box, frame_count, i, output_dir):
    """Save the cropped vehicle image."""
    x1, y1, x2, y2 = map(int, box)
    cropped_vehicle = frame[y1:y2, x1:x2]
    crop_filename = os.path.join(output_dir, f'vehicle_{frame_count}_{i}.jpg')
    cv2.imwrite(crop_filename, cropped_vehicle)
    print(f"Saved cropped vehicle to {crop_filename}")
    return crop_filename, cropped_vehicle
