import subprocess
import numpy as np
import cv2

def start_ffmpeg_stream(rtmp_url):
    """Start FFmpeg process to pipe the RTMP stream as raw frames."""
    ffmpeg_command = [
        'ffmpeg',
        '-i', rtmp_url,
        '-f', 'rawvideo',
        '-pix_fmt', 'bgr24',
        '-vcodec', 'rawvideo',
        '-an',
        '-sn',
        '-r', '30',
        '-'
    ]
    return subprocess.Popen(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

def read_frame(process, frame_size, width, height):
    """Read a single frame from the RTMP stream."""
    raw_frame = process.stdout.read(frame_size)
    if len(raw_frame) != frame_size:
        print("Error: Unable to read frame")
        return None
    return np.frombuffer(raw_frame, np.uint8).reshape((height, width, 3))

# Function to draw lane ROI on the frame
def draw_lane_roi(frame, lane_boundary_points):
    """Draw the region of interest (ROI) on the frame."""
    cv2.polylines(frame, [lane_boundary_points], isClosed=True, color=(0, 255, 0), thickness=2)
    return frame

# Function to calculate the centroid of a polygon
def calculate_polygon_centroid(points):
    """Calculate the centroid of a polygon defined by lane boundary points."""
    moments = cv2.moments(points)
    if moments["m00"] != 0:
        cX = int(moments["m10"] / moments["m00"])
        cY = int(moments["m01"] / moments["m00"])
        return (cX, cY)
    else:
        return None

# Function to check if vehicle's center is near the ROI centroid
def is_vehicle_in_roi(roi_centroid, vehicle_box, threshold=50):
    """Check if the vehicle's center is within a certain threshold of the ROI centroid."""
    x1, y1, x2, y2 = map(int, vehicle_box)
    vehicle_center = ((x1 + x2) // 2, (y1 + y2) // 2)
    
    if roi_centroid:
        distance = np.linalg.norm(np.array(vehicle_center) - np.array(roi_centroid))
        if distance < threshold:
            return True
    return False
