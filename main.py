from datetime import datetime
from ultralytics import YOLO
import cv2
from utils import stream_utils, vehicle_detection, number_plate_detection, utils, ocr_detection
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

# Global variables for configuration
rtmp_url = "rtmp://13.201.90.128/live"
output_dir = "./detected_vehicles"
number_plate_dir = "./plates"
final_detected_dir = "./FINAL_DETECTED_VEHICLES"
csv_file_path = './detected_vehicles.csv'
width, height = 1920, 1080
confidence_threshold = 0.5
skip_frames = 5

def create_directories(*dirs):
    """Create multiple directories if they do not already exist."""
    for directory in dirs:
        if not os.path.exists(directory):
            os.makedirs(directory)

# Create necessary directories
create_directories(output_dir, number_plate_dir, final_detected_dir)

# Initialize YOLO models for vehicle and plate detection
vehicle_model = YOLO("./models/best.pt")  # Path to vehicle detection model
plate_model = YOLO("./models/license_plate_detector.pt")  # Path to license plate detection model

# Define lane boundary points
lane_boundary_points = np.array([[250, 500], [750, 500], [1300, 800], [400, 800]], np.int32)
lane_boundary_points = lane_boundary_points.reshape((-1, 1, 2))

# Calculate the centroid of the lane ROI polygon
roi_centroid = stream_utils.calculate_polygon_centroid(lane_boundary_points)

def process_vehicle_data(crop_filename, number_plate_image_path, frame_count, detected_text, detected_class):
    """Process vehicle data and return a dictionary."""
    vehicle_folder_path = os.path.join(final_detected_dir, detected_text)
    os.makedirs(vehicle_folder_path, exist_ok=True)

    # Move the vehicle and number plate images to the folder
    final_vehicle_path = os.path.join(vehicle_folder_path, f"vehicle_{frame_count}.jpg")
    final_plate_path = os.path.join(vehicle_folder_path, f"plate_{frame_count}.jpg")
    
    os.rename(crop_filename, final_vehicle_path)
    os.rename(number_plate_image_path, final_plate_path)

    # Prepare the vehicle data
    vehicle_data = {
        'Vehicle Image Path': final_vehicle_path,
        'Number Plate Image Path': final_plate_path,
        'Detected Number': detected_text,
        'Time Detected': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'Detected Class': detected_class
    }
    
    return vehicle_data

def process_stream():
    """Main function to process the RTMP stream, detect vehicles, and save data."""
    frame_size = width * height * 3
    process = stream_utils.start_ffmpeg_stream(rtmp_url)
    vehicle_data_list = []
    frame_count = 0

    with ThreadPoolExecutor() as executor:
        futures = []

        while True:
            frame = stream_utils.read_frame(process, frame_size, width, height)
            if frame is None:
                break

            frame_count += 1

            # Skip frames to reduce lag
            if frame_count % skip_frames != 0:
                continue

            # Draw lane ROI on the current frame
            frame = stream_utils.draw_lane_roi(frame, lane_boundary_points)

            results = vehicle_detection.detect_vehicles(frame, vehicle_model)

            # Process each vehicle detection
            for result in results:
                for i, box in enumerate(result.boxes.xyxy):
                    conf = result.boxes.conf[i]
                    detected_class = vehicle_model.names[int(result.boxes.cls[i])]

                    # Check if vehicle detection confidence is above threshold
                    if conf > confidence_threshold:
                        
                        # Check if the vehicle is within the ROI center
                        if stream_utils.is_vehicle_in_roi(roi_centroid, box):
                            print(f"Vehicle reached ROI center, saving... Frame: {frame_count}, Vehicle: {i}")
                            
                            # Save cropped vehicle image and detect number plate
                            crop_filename, cropped_vehicle = vehicle_detection.save_vehicle_image(frame, box, frame_count, i, output_dir)
                            number_plate_image_path = number_plate_detection.detect_number_plate(cropped_vehicle, plate_model, frame_count, i, number_plate_dir)

                            # If a number plate image is detected, perform OCR and save
                            if number_plate_image_path:
                                detected_text = ocr_detection.perform_ocr(number_plate_image_path)

                                if detected_text:
                                    # Submit the vehicle data processing to the executor
                                    futures.append(executor.submit(process_vehicle_data, crop_filename, number_plate_image_path, frame_count, detected_text, detected_class))

            # Display the frame with detections and ROI
            annotated_frame = result.plot()
            cv2.imshow('RTMP Stream with YOLOv8 Detections and ROI', annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Wait for all futures to complete and gather vehicle data
        for future in as_completed(futures):
            vehicle_data = future.result()
            vehicle_data_list.append(vehicle_data)

    # Terminate the FFmpeg process
    process.terminate()
    cv2.destroyAllWindows()

    # Save vehicle data to CSV
    utils.save_data_to_csv(vehicle_data_list, csv_file_path)

if __name__ == "__main__":
    process_stream()
