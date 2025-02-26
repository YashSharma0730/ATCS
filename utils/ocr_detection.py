import cv2
import easyocr
import numpy as np

def perform_ocr(image_path):
    """Perform OCR on the given image and return the detected text."""
    # Load the image
    image = cv2.imread("/Users/yashsharma/Desktop/ATCS_FINAL/typesofcarnumberplates-02-01.jpg")

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Increase contrast
    alpha = 1.5  # Simple contrast control
    beta = 0     # Simple brightness control
    contrasted = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)

    # Sharpen the image using a kernel
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(contrasted, -1, kernel)

    # Resize the image to make the text clearer (scale up)
    resized = cv2.resize(sharpened, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

    # Apply adaptive thresholding
    threshold = cv2.adaptiveThreshold(resized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Perform OCR using EasyOCR
    reader = easyocr.Reader(['en'])
    result_easyocr = reader.readtext(threshold)

    # Extract the detected text
    detected_text = ' '.join([res[1] for res in result_easyocr])
    print(f"Detected Text: {detected_text}")

    return detected_text
