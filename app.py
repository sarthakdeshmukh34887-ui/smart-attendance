import os
import streamlit as st
import cv2
import numpy as np
import pickle
import av
import threading
import pandas as pd
import serial
import serial.tools.list_ports
import time
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from datetime import datetime
import tensorflow as tf
from face_utils import create_embedder, extract_face_crops, get_embedding
from train_model import retrain_system

# --- HIDE TENSORFLOW LOGS ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# --- PAGE CONFIG ---
st.set_page_config(page_title="Biometric Attendance Dashboard", page_icon="🛡️", layout="wide")

# --- SUCCESS BEEP (JavaScript) ---
def play_beep():
    beep_html = """
        <audio autoplay>
            <source src="https://cdn.pixabay.com/audio/2022/03/15/audio_507204595e.mp3" type="audio/mpeg">
        </audio>
    """
    st.components.v1.html(beep_html, height=0)

# --- CONSTANTS ---
MODEL_PATH = 'models/attendance_model.h5'
LABEL_PATH = 'models/label_map.pkl'
FINGER_MAP_PATH = 'models/finger_map.pkl'
CSV_FILE = "attendance.csv"
lock = threading.Lock()

# ==========================================
# 1. LOAD AI RESOURCES
# ==========================================
@st.cache_resource
def load_resources():
    embedder = create_embedder()
    model, label_map, finger_map = None, {}, {}
    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
    if os.path.exists(LABEL_PATH):
        with open(LABEL_PATH, 'rb') as f: label_map = pickle.load(f)
    if os.path.exists(FINGER_MAP_PATH):
        with open(FINGER_MAP_PATH, 'rb') as f: finger_map = pickle.load(f)
    return embedder, model, label_map, finger_map

embedder, model, label_map, finger_map = load_resources()

# ==========================================
# 2. ATTENDANCE LOGIC
# ==========================================
def mark_attendance(roll_name, method):
    try:
        roll, name = roll_name.split('_', 1)
    except:
        roll, name = "??", roll_name
    
    with lock:
        now = datetime.now()
        df = pd.DataFrame({
            "Date": [now.strftime("%Y-%m-%d")],
            "Time": [now.strftime("%H:%M:%S")],
            "Roll": [roll], "Name": [name],
            "Method": [method], "Status": ["Present"]
        })
        file_exists = os.path.exists(CSV_FILE)
        df.to_csv(CSV_FILE, mode='a', header=not file_exists, index=False)

# ==========================================
# 3. VIDEO PROCESSOR CLASS
# ==========================================
class DualProcessor(VideoProcessorBase):
    def __init__(self, mode="mark"):
        self.mode = mode
        self.face_samples = []
        self.marked = set()

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        faces = extract_face_crops(img)

        for (x, y, w, h), crop in faces:
            emb = get_embedding(crop, embedder)
            if emb is not None:
                if self.mode == "register":
                    if len(self.face_samples) < 20:
                        self.face_samples.append(emb)
                    color, text = (0, 255, 255), f"Samples: {len(self.face_samples)}/20"
                else:
                    if model:
                        pred = model.predict(np.expand_dims(emb, axis=0), verbose=0)
                        if np.max(pred) > 0.8:
                            lbl = label_map.get(np.argmax(pred), "Unknown")
                            if lbl != "Unknown" and lbl not in self.marked:
                                mark_attendance(lbl, "Face")
                                self.marked.add(lbl)
                            color, text = (0, 255, 0), lbl.split('_')[-1]
                        else:
                            color, text = (0, 0, 255), "Unknown"
                
                cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
                cv2.putText(img, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ==========================================
# 4. HARDWARE SETTINGS (HEARTBEAT FIX)
# ==========================================
st.sidebar.title("🔌 Hardware Settings")
ports = [port.device for port in serial.tools.list_ports.comports()]
default_idx = 0
if "COM3" in ports: default_idx = ports.index("COM3")
elif "COM4" in ports: default_idx = ports.index("COM4")

selected_port = st.sidebar.selectbox("Select Port", ports if ports else ["None"], index=default_idx)

@st.cache_resource
def get_serial(port):
    if port == "None": return None
    try:
        s = serial.Serial(port, 9600, timeout=0.1)
        time.sleep(1) # Allow port to settle
        return s
    except: return None

ser = get_serial(selected_port)

if st.sidebar.button("🔄 Refresh Hardware"):
    st.cache_resource.clear()
    st.rerun()

if ser: st.success(f"✅ Arduino R307 detected on **{selected_port}**")
else: st.error("❌ Arduino not detected. Check COM Port.")

# ==========================================
# 5. NAVIGATION & UI
# ==========================================
menu = st.sidebar.selectbox("Navigation", ["Mark Attendance", "Register New User", "View Records"])

if menu == "Register New User":
    st.header("👤 Biometric Enrollment Center")
    tab_face, tab_finger = st.tabs(["📸 Face Registration", "☝️ Fingerprint Registration"])

    with tab_face:
        st.subheader("Face Enrollment")
        c1, c2 = st.columns(2)
        f_name, f_roll = c1.text_input("Name", key="fn"), c2.text_input("Roll", key="fr")
        face_ctx = webrtc_streamer(key="face_reg", video_processor_factory=lambda: DualProcessor(mode="register"), media_stream_constraints={"video": True, "audio": False})
        if st.button("💾 Save Face Samples"):
            if face_ctx.video_processor and len(face_ctx.video_processor.face_samples) >= 20:
                label = f"{f_roll}_{f_name}"
                with open(f"data/{label}.pkl", 'wb') as f: pickle.dump(face_ctx.video_processor.face_samples, f)
                with st.spinner("Retraining..."): retrain_system()
                st.success("Face Registered!")
                play_beep()

    with tab_finger:
        st.subheader("Fingerprint Enrollment")
        if not ser: st.warning("Please connect Arduino first.")
        else:
            c1, c2, c3 = st.columns(3)
            g_name, g_roll, g_id = c1.text_input("Name", key="gn"), c2.text_input("Roll", key="gr"), c3.number_input("Slot", 1, 127, key="gi")
            finger_status = st.empty()
            if st.button("🔴 Start Fingerprint Scan"):
                ser.write(f"ENROLL:{g_id}\n".encode())
                timeout = time.time() + 30
                while time.time() < timeout:
                    if ser.in_waiting:
                        msg = ser.readline().decode(errors='ignore').strip()
                        if "PLACE_FINGER" in msg: finger_status.warning("☝️ PLACE FINGER")
                        if "REMOVE_FINGER" in msg: finger_status.warning("👋 REMOVE FINGER")
                        if "ENROLL_SUCCESS" in msg:
                            finger_map[str(g_id)] = f"{g_roll}_{g_name}"
                            with open(FINGER_MAP_PATH, 'wb') as f: pickle.dump(finger_map, f)
                            finger_status.success(f"✅ Slot {g_id} Enrolled!")
                            play_beep()
                            break
                    time.sleep(0.1)

elif menu == "Mark Attendance":
    st.header("📷 Live Attendance Terminal")
    
    # Background Fingerprint Listener
    if ser and ser.in_waiting:
        line = ser.readline().decode(errors='ignore').strip()
        if "FOUND_ID:" in line:
            fid = line.split(":")[-1]
            if fid in finger_map:
                mark_attendance(finger_map[fid], "Fingerprint")
                st.toast(f"☝️ Fingerprint Match: {finger_map[fid]}", icon="✅")
                play_beep()

    webrtc_streamer(key="att", video_processor_factory=lambda: DualProcessor(mode="mark"), media_stream_constraints={"video": True, "audio": False})

elif menu == "View Records":
    st.header("📊 Attendance Log")
    if os.path.exists(CSV_FILE):
        df = pd.read_csv(CSV_FILE, on_bad_lines='skip')
        st.dataframe(df.iloc[::-1], use_container_width=True)
        if st.button("🗑️ Clear Records"):
            os.remove(CSV_FILE)
            st.rerun()
    else: st.info("No records found.")