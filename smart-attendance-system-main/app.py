import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Hide Info and Warning logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Hide oneDNN custom operations message



import streamlit as st
import cv2
import numpy as np
import os
import pickle
import av
import threading
import pandas as pd
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from datetime import datetime
import tensorflow as tf
from face_utils import create_embedder, extract_face_crops, get_embedding
from train_model import retrain_system

# Page Config
st.set_page_config(page_title="Smart Attendance System", page_icon="👤", layout="wide")

# --- GLOBAL SETUP ---
MODEL_PATH = os.path.join('models', 'attendance_model.h5')
LABEL_PATH = os.path.join('models', 'label_map.pkl')
CSV_FILE = "attendance.csv"

if not os.path.exists('data'): os.makedirs('data')

# Thread Lock to prevent file corruption when writing to CSV
lock = threading.Lock()

# Load AI Models
@st.cache_resource
def load_models():
    embedder = create_embedder()
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        with open(LABEL_PATH, 'rb') as f:
            label_map = pickle.load(f)
    except:
        model = None
        label_map = {}
    return embedder, model, label_map

embedder, model, label_map = load_models()

# --- SIDEBAR ---
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose Mode", ["Register Student", "Mark Attendance", "View Attendance Data"])

# ==========================================
# 1. REGISTRATION PROCESSOR
# ==========================================
class RegistrationProcessor(VideoProcessorBase):
    def __init__(self):
        self.captured_embeddings = []
    
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        
        if len(self.captured_embeddings) < 20:
            faces = extract_face_crops(img)
            if faces:
                (x, y, w, h), face_crop = faces[0]
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                emb = get_embedding(face_crop, embedder)
                if emb is not None:
                    self.captured_embeddings.append(emb)
        
        cv2.putText(img, f"Captured: {len(self.captured_embeddings)}/20", (30, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ==========================================
# 2. ATTENDANCE PROCESSOR (NOW WRITES TO CSV!)
# ==========================================
class AttendanceProcessor(VideoProcessorBase):
    def __init__(self):
        # Keep track of who is marked in THIS session
        self.marked_session = set()

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        
        if model is None:
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        faces_found = extract_face_crops(img)
        
        for (x, y, w, h), face_crop in faces_found:
            emb = get_embedding(face_crop, embedder)
            
            if emb is not None:
                prediction = model.predict(np.expand_dims(emb, axis=0), verbose=0)
                max_prob = np.max(prediction)
                class_id = np.argmax(prediction)
                
                if max_prob > 0.80:
                    student_label = label_map.get(class_id, "Unknown")
                    
                    # Split name carefully
                    try:
                        roll, name = student_label.split('_', 1)
                    except:
                        roll, name = "??", student_label

                    color = (0, 255, 0)
                    text = f"{name} {int(max_prob*100)}%"

                    # --- CSV WRITING LOGIC ---
                    if student_label not in self.marked_session:
                        with lock: # Prevent crashing if two people enter at once
                            now = datetime.now()
                            date_str = now.strftime("%Y-%m-%d")
                            time_str = now.strftime("%H:%M:%S")
                            
                            new_data = pd.DataFrame({
                                "Date": [date_str], 
                                "Time": [time_str], 
                                "Roll": [roll], 
                                "Name": [name], 
                                "Status": ["Present"]
                            })

                            # Append to CSV
                            if not os.path.exists(CSV_FILE):
                                new_data.to_csv(CSV_FILE, index=False)
                            else:
                                new_data.to_csv(CSV_FILE, mode='a', header=False, index=False)
                            
                            print(f"✅ Marked: {name} at {time_str}")
                        
                        self.marked_session.add(student_label)
                    # -------------------------
                    
                else:
                    color = (0, 0, 255)
                    text = "Unknown"
                
                cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
                cv2.putText(img, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ==========================================
# APP UI
# ==========================================
if app_mode == "Register Student":
    st.title("👤 New Student Registration")
    col1, col2 = st.columns(2)
    name = col1.text_input("Name")
    roll = col2.text_input("Roll No")

    ctx = webrtc_streamer(
        key="registration", 
        video_processor_factory=RegistrationProcessor,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False}
    )

    if st.button("💾 Save Data"):
        if ctx.video_processor:
            captured_data = ctx.video_processor.captured_embeddings
            if len(captured_data) < 20:
                st.warning(f"Only captured {len(captured_data)}/20 frames. Keep scanning!")
            else:
                filename = f"data/{roll}_{name}.pkl"
                with open(filename, 'wb') as f:
                    pickle.dump(captured_data, f)
                st.success("✅ Data Saved! Retraining model...")
                retrain_system()
                st.success("🚀 System Updated!")

elif app_mode == "Mark Attendance":
    st.title("📷 Live Attendance Monitor")
    
    if model is None:
        st.error("❌ No AI Model found. Please register a student first.")
    else:
        st.write("System is running... Attendance will auto-save to CSV.")
        webrtc_streamer(
            key="attendance",
            video_processor_factory=AttendanceProcessor,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            media_stream_constraints={"video": True, "audio": False}
        )

elif app_mode == "View Attendance Data":
    st.title("📊 Attendance Records")
    
    # Refresh button to reload data manually
    if st.button("Refresh Data"):
        st.rerun()

    if os.path.exists(CSV_FILE):
        df = pd.read_csv(CSV_FILE)
        # Show most recent entries at the top
        st.dataframe(df.iloc[::-1]) 
    else:
        st.info("No attendance records found yet.")