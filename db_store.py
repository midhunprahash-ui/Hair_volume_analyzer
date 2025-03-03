import cv2
import torch
import numpy as np
import mysql.connector
import datetime
from ultralytics import YOLO


model = YOLO("best.pt")  


def connect_db():
    return mysql.connector.connect(
        host="localhost",
        user="root",  
        password="",  
        database="hair_analysis"
    )


def create_table():
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS hair_data (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        date DATE,
                        time TIME,
                        volume FLOAT
                      )''')
    conn.commit()
    conn.close()


def store_hair_volume(volume):
    conn = connect_db()
    cursor = conn.cursor()
    

    create_table()


    now = datetime.datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")


    cursor.execute("INSERT INTO hair_data (date, time, volume) VALUES (%s, %s, %s)", (date, time, volume))
    conn.commit()
    conn.close()


def analyze_hair_trend(current_volume):
    conn = connect_db()
    cursor = conn.cursor()

    cursor.execute("SELECT volume FROM hair_data ORDER BY id DESC LIMIT 1")
    last_record = cursor.fetchone()

    conn.close()

    if last_record:
        last_volume = last_record[0]
        if current_volume > last_volume:
            return "Hair Volume Improved! ✅"
        elif current_volume < last_volume:
            return "Slight Hair Loss Detected ❌"
        else:
            return "No Change in Hair Volume 🔄"
    else:
        return "First Record - No Comparison Available"

# Start webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO inference
    results = model(frame)

    # Check if masks exist
    if results[0].masks is not None and len(results[0].masks.data) > 0:
        mask = results[0].masks.data[0].cpu().numpy() * 255  # Convert to NumPy and scale

        # Resize mask to match frame size
        mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))

        # Convert single-channel mask to 3-channel (green color)
        green_mask = np.zeros_like(frame)
        green_mask[:, :, 1] = mask_resized  # Assign resized mask to green channel

        # Overlay mask on frame
        frame = cv2.addWeighted(frame, 1, green_mask, 0.5, 0)

        # Hair volume calculation (percentage of hair pixels in mask)
        hair_pixels = np.count_nonzero(mask_resized)
        total_pixels = mask_resized.size
        hair_volume = round((hair_pixels / total_pixels) * 100, 2)  # Scale to 1-100%

        # Store hair volume in database
        store_hair_volume(hair_volume)

        # Compare with previous records
        trend_message = analyze_hair_trend(hair_volume)

        # Display hair volume percentage
        text = f"Hair Volume: {hair_volume}%"
        cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Display trend analysis
        cv2.putText(frame, trend_message, (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show output
    cv2.imshow("Hair Segmentation", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()