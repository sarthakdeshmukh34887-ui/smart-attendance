import cv2
import numpy as np
import tensorflow as tf
import pickle
import pandas as pd
import os
from datetime import datetime
from face_utils import create_embedder, extract_face_crops, get_embedding

# Config
MODEL_PATH = os.path.join('models', 'attendance_model.h5')
LABEL_PATH = os.path.join('models', 'label_map.pkl')
CSV_FILE = "attendance.csv"

# Check if Training has been done
if not os.path.exists(MODEL_PATH):
    print("❌ Model not found! Run 'register.py' first.")
    exit()

# Load AI
print("⏳ Loading AI Model...")
model = tf.keras.models.load_model(MODEL_PATH)
with open(LABEL_PATH, 'rb') as f:
    label_map = pickle.load(f)

embedder = create_embedder()
session_attendance = set()

def mark_attendance(student_label):
    # --- FIX STARTS HERE ---
    # We use split('_', 1) to split only on the FIRST underscore.
    # This handles names like "Om_Prakash" correctly.
    try:
        roll, name = student_label.split('_', 1)
    except ValueError:
        # If the format is completely wrong, just use the whole label as the name
        roll = "Unknown"
        name = student_label
    # --- FIX ENDS HERE ---

    if student_label not in session_attendance:
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")
        print(f"✅ MARKED: {name}")
        
        data = {"Date": [date_str], "Time": [time_str], "Roll": [roll], "Name": [name], "Status": ["Present"]}
        df = pd.DataFrame(data)
        
        if not os.path.exists(CSV_FILE):
            df.to_csv(CSV_FILE, index=False)
        else:
            df.to_csv(CSV_FILE, mode='a', header=False, index=False)
            
        session_attendance.add(student_label)

# Start Camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break
    
    # 1. Detect ALL Faces
    faces_found = extract_face_crops(frame)
    
    for (x, y, w, h), face_crop in faces_found:
        
        # 2. Get Embedding for THIS face
        emb = get_embedding(face_crop, embedder)
        
        if emb is not None:
            # 3. Predict
            prediction = model.predict(np.expand_dims(emb, axis=0), verbose=0)
            max_prob = np.max(prediction)
            class_id = np.argmax(prediction)
            
            # 4. Draw Dynamic Box & Label
            if max_prob > 0.80:
                student_label = label_map[class_id] # Get the full label (e.g. 23_Om)
                color = (0, 255, 0) # Green
                
                # Extract name safely for display
                try:
                    display_name = student_label.split('_', 1)[1]
                except:
                    display_name = student_label
                    
                text = f"{display_name} {int(max_prob*100)}%"
                mark_attendance(student_label)
            else:
                color = (0, 0, 255) # Red
                text = "Unknown"
            
            # Draw the box at (x, y) with width w and height h
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("Smart Attendance", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()