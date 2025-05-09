import cv2
import torch
import numpy as np
from ultralytics import YOLO

# Load trained model
model = YOLO("best.pt")

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Perform inference
    results = model(frame)

    # Get total pixels in the frame
    total_pixels = frame.shape[0] * frame.shape[1]  

    # Check if masks exist
    if results[0].masks is not None:
        mask = results[0].masks.data[0].cpu().numpy()  # Convert to NumPy
        mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))  # Resize to match frame

        # Convert mask to binary (0 or 255)
        mask = (mask > 0.5).astype("uint8") * 255  

        # Apply Morphological Transformations to clean noise
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Close small holes
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # Remove noise

        # Find contours of the hair region
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Ensure the largest contour is considered
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            hair_pixels = cv2.contourArea(largest_contour)  # Use contour area instead of counting pixels
        else:
            hair_pixels = 0

        # Scale hair volume from 1 to 100%
        hair_volume = max(1, min(100, int((hair_pixels / total_pixels) * 100)))  

        # Create a green mask (0, 255, 0) in BGR format
        green_mask = cv2.merge([mask * 0, mask, mask * 0])  

        # Overlay green mask on frame
        overlay = cv2.addWeighted(frame, 1, green_mask.astype("uint8"), 0.5, 0)
    else:
        print("No segmentation detected")
        overlay = frame  # Show original frame if no segmentation
        hair_volume = 1  # Minimum volume is 1%

    # Display hair volume percentage on frame in RED color
    text = f"Hair Volume: {hair_volume}%"
    cv2.putText(overlay, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Show output
    cv2.imshow("Hair Segmentation - Green Mask", overlay)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()