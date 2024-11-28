import os
import pandas as pd
import cv2
from datetime import datetime
from flask import Flask, request, render_template, redirect, url_for

app = Flask(__name__)

EXCEL_FILE = 'attendance.xlsx'
IMAGE_FOLDER = 'images'

# Ensure the images folder exists
os.makedirs(IMAGE_FOLDER, exist_ok=True)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    name = request.form['name']
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    image_path = os.path.join(IMAGE_FOLDER, f"{name}_{timestamp.replace(':', '-')}.jpg")

    # Capture the face photo and save it
    if capture_face(image_path):
        try:
            # Initialize the Excel file with headers if it doesn’t exist
            if not os.path.exists(EXCEL_FILE):
                pd.DataFrame(columns=['Name', 'Time', 'Photo']).to_excel(EXCEL_FILE, index=False, engine='openpyxl')

            # Load the existing data
            df = pd.read_excel(EXCEL_FILE, engine='openpyxl')

            # Add the new entry
            new_entry = pd.DataFrame([[name, timestamp, image_path]], columns=['Name', 'Time', 'Photo'])
            updated_df = pd.concat([df, new_entry], ignore_index=True)

            # Save to Excel
            updated_df.to_excel(EXCEL_FILE, index=False, engine='openpyxl')

            return f"Attendance recorded for {name} at {timestamp} with face photo saved."
        
        except PermissionError:
            return "Error: Unable to write to 'attendance.xlsx'. Please ensure the file is closed and try again."

    else:
        return "Face capture failed. Please try again."


def capture_face(image_path):
    """Capture a face photo using the webcam and save it."""
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) > 0:
            for (x, y, w, h) in faces:
                face = frame[y:y+h, x:x+w]
                cv2.imwrite(image_path, face)  # Save the face photo
                cap.release()
                cv2.destroyAllWindows()
                return True  # Capture successful

        cv2.imshow("Capturing Face", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return False  # Capture failed

if __name__ == '__main__':
    app.run(debug=True)
