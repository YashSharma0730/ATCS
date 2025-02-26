import os
import pandas as pd

def save_data_to_csv(vehicle_data, csv_file_path):
    """Save vehicle detection data to CSV."""
    vehicle_df = pd.DataFrame(vehicle_data)
    vehicle_df.to_csv(csv_file_path, index=False)
    print(f"Detected vehicle data saved to {csv_file_path}")

def create_directories(output_dir, number_plate_dir):
    """Create necessary directories."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(number_plate_dir, exist_ok=True)
