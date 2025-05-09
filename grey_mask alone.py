import cv2
import torch
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

    # Check if masks exist
    if results[0].masks is not None:
        mask = results[0].masks.data[0].cpu().numpy() * 255  # Convert to NumPy and scale

        # Ensure mask is the same size as frame
        mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))

        # Convert grayscale mask to 3 channels if needed
        if len(mask.shape) == 2:
            mask = cv2.cvtColor(mask.astype("uint8"), cv2.COLOR_GRAY2BGR)

        # Ensure data type consistency
        frame = frame.astype("uint8")
        mask = mask.astype("uint8")

        # Overlay mask on frame
        overlay = cv2.addWeighted(frame, 1, mask, 0.5, 0)
    else:
        print("No segmentation detected")
        overlay = frame  # Show original frame if no segmentation

    # Show output
    cv2.imshow("Hair Segmentation", overlay)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
