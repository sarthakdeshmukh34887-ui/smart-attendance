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
from scipy.spatial.distance import cosine
from face_utils import create_embedder, extract_face_crops, get_embedding
from train_model import retrain_system

# --- HIDE TENSORFLOW LOGS ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# --- PAGE CONFIG ---
st.set_page_config(page_title="Biometric Attendance Dashboard", page_icon="🛡️", layout="wide")

# --- SUCCESS BEEP ---
def play_beep():
    beep_html = """
        <audio autoplay>
            <source src="https://cdn.pixabay.com/audio/2022/03/15/audio_507204595e.mp3" type="audio/mpeg">
        </audio>
    """
    st.components.v1.html(beep_html, height=0)

# --- CONSTANTS ---
MODEL_PATH      = 'models/attendance_model.h5'
LABEL_PATH      = 'models/label_map.pkl'
FINGER_MAP_PATH = 'models/finger_map.pkl'
DATA_DIR        = 'data'
CSV_FILE        = "attendance.csv"
lock            = threading.Lock()

# ── Thresholds ────────────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.45
ENTROPY_THRESHOLD    = 1.50
COSINE_SIM_THRESHOLD = 0.20

# ── Face enhancement — fixes backlit / dark faces ─────────────────────────────
def enhance_face(img):
    """CLAHE on L channel — auto-corrects dark/backlit face crops."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

# ── Gate helper functions ─────────────────────────────────────────────────────
def softmax_entropy(probs):
    probs = np.clip(probs, 1e-9, 1.0)
    return float(-np.sum(probs * np.log(probs)))

def best_cosine_sim(embedding, class_id, embedding_store):
    stored = embedding_store.get(class_id, [])
    if not stored:
        return 0.0
    return max(1 - cosine(embedding, s) for s in stored)

def is_known_face(emb, probs, class_id, embedding_store):
    confidence = float(np.max(probs))
    entropy    = softmax_entropy(probs)
    cos_sim    = best_cosine_sim(emb, class_id, embedding_store)
    known = (
        confidence >= CONFIDENCE_THRESHOLD and
        entropy    <= ENTROPY_THRESHOLD    and
        cos_sim    >= COSINE_SIM_THRESHOLD
    )
    return known, confidence, entropy, cos_sim

# ==========================================
# 1. LOAD AI RESOURCES
# ==========================================
@st.cache_resource
def load_resources():
    embedder = create_embedder()
    model, label_map, finger_map, embedding_store = None, {}, {}, {}

    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
    if os.path.exists(LABEL_PATH):
        with open(LABEL_PATH, 'rb') as f:
            label_map = pickle.load(f)

    for idx, name in label_map.items():
        pkl_path = os.path.join(DATA_DIR, f'{name}.pkl')
        if os.path.exists(pkl_path):
            with open(pkl_path, 'rb') as f:
                embedding_store[idx] = pickle.load(f)

    if os.path.exists(FINGER_MAP_PATH):
        with open(FINGER_MAP_PATH, 'rb') as f:
            finger_map = pickle.load(f)

    return embedder, model, label_map, finger_map, embedding_store

embedder, model, label_map, finger_map, embedding_store = load_resources()

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
            "Date":   [now.strftime("%Y-%m-%d")],
            "Time":   [now.strftime("%H:%M:%S")],
            "Roll":   [roll],
            "Name":   [name],
            "Method": [method],
            "Status": ["Present"],
        })
        file_exists = os.path.exists(CSV_FILE)
        df.to_csv(CSV_FILE, mode='a', header=not file_exists, index=False)

# ==========================================
# 3. VIDEO PROCESSOR CLASS
# ==========================================
class DualProcessor(VideoProcessorBase):
    def __init__(self, mode="mark"):
        self.mode         = mode
        self.face_samples = []
        self.marked       = set()

    def recv(self, frame):
        img   = frame.to_ndarray(format="bgr24")
        img   = cv2.flip(img, 1)
        faces = extract_face_crops(img)

        for (x, y, w, h), crop in faces:
            # enhance BEFORE embedding — fixes dark/backlit faces
            crop = enhance_face(crop)
            emb  = get_embedding(crop, embedder)
            if emb is None:
                continue

            if self.mode == "register":
                if len(self.face_samples) < 50:
                    self.face_samples.append(emb)
                color = (0, 255, 255)
                text  = f"Samples: {len(self.face_samples)}/50"

            else:
                if model is None:
                    color, text = (0, 0, 255), "No model"
                else:
                    norm     = np.linalg.norm(emb)
                    emb_norm = emb / max(norm, 1e-8)

                    probs    = model.predict(
                        np.expand_dims(emb_norm, axis=0), verbose=0
                    )[0]
                    class_id = int(np.argmax(probs))

                    known, conf, entropy, cos_sim = is_known_face(
                        emb_norm, probs, class_id, embedding_store
                    )

                    if known:
                        lbl = label_map.get(class_id, "Unknown")
                        if lbl not in self.marked:
                            mark_attendance(lbl, "Face")
                            self.marked.add(lbl)
                        color = (0, 255, 0)
                        text  = f"{lbl.split('_')[-1]} {int(conf*100)}%"
                    else:
                        color = (0, 0, 255)
                        text  = "Unknown"

            cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
            cv2.putText(img, text, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ==========================================
# 4. HARDWARE SETTINGS
# ==========================================
st.sidebar.title("🔌 Hardware Settings")
ports       = [port.device for port in serial.tools.list_ports.comports()]
default_idx = 0
if "COM3" in ports:   default_idx = ports.index("COM3")
elif "COM4" in ports: default_idx = ports.index("COM4")

selected_port = st.sidebar.selectbox(
    "Select Port", ports if ports else ["None"], index=default_idx
)

@st.cache_resource
def get_serial(port):
    if port == "None": return None
    try:
        s = serial.Serial(port, 9600, timeout=0.1)
        time.sleep(1)
        return s
    except:
        return None

ser = get_serial(selected_port)

if st.sidebar.button("🔄 Refresh Hardware"):
    st.cache_resource.clear()
    st.rerun()

if ser: st.success(f"✅ Arduino R307 detected on **{selected_port}**")
else:   st.error("❌ Arduino not detected. Check COM Port.")

# ==========================================
# 5. NAVIGATION & UI
# ==========================================
menu = st.sidebar.selectbox(
    "Navigation", ["Mark Attendance", "Register New User", "View Records"]
)

# ── Register ──────────────────────────────────────────────────────────────────
if menu == "Register New User":
    st.header("👤 Biometric Enrollment Center")
    tab_face, tab_finger = st.tabs(["📸 Face Registration", "☝️ Fingerprint Registration"])

    with tab_face:
        st.subheader("Face Enrollment")
        st.info("💡 Tip: Register in the same lighting where attendance will be taken. Move your head slightly during registration for better accuracy.")
        c1, c2   = st.columns(2)
        f_name   = c1.text_input("Name", key="fn")
        f_roll   = c2.text_input("Roll", key="fr")
        face_ctx = webrtc_streamer(
            key="face_reg",
            video_processor_factory=lambda: DualProcessor(mode="register"),
            media_stream_constraints={"video": True, "audio": False}
        )
        if st.button("💾 Save Face Samples"):
            if face_ctx.video_processor and len(face_ctx.video_processor.face_samples) >= 50:
                label = f"{f_roll}_{f_name}"
                with open(f"data/{label}.pkl", 'wb') as f:
                    pickle.dump(face_ctx.video_processor.face_samples, f)
                with st.spinner("🔄 Retraining model... please wait"):
                    success = retrain_system()
                if success:
                    st.cache_resource.clear()
                    st.success(f"✅ {f_name} registered and model updated!")
                    play_beep()
                    st.balloons()
                else:
                    st.error("❌ Retraining failed — need at least 2 registered students.")
            else:
                samples = len(face_ctx.video_processor.face_samples) if face_ctx.video_processor else 0
                st.warning(f"⚠️ Not enough samples yet ({samples}/50). Keep your face in frame.")

    with tab_finger:
        st.subheader("Fingerprint Enrollment")
        if not ser:
            st.warning("Please connect Arduino first.")
        else:
            c1, c2, c3 = st.columns(3)
            g_name  = c1.text_input("Name", key="gn")
            g_roll  = c2.text_input("Roll", key="gr")
            g_id    = c3.number_input("Slot", 1, 127, key="gi")
            finger_status = st.empty()
            if st.button("🔴 Start Fingerprint Scan"):
                ser.write(f"ENROLL:{g_id}\n".encode())
                timeout = time.time() + 30
                while time.time() < timeout:
                    if ser.in_waiting:
                        msg = ser.readline().decode(errors='ignore').strip()
                        if "PLACE_FINGER"   in msg: finger_status.warning("☝️ PLACE FINGER")
                        if "REMOVE_FINGER"  in msg: finger_status.warning("👋 REMOVE FINGER")
                        if "ENROLL_SUCCESS" in msg:
                            finger_map[str(g_id)] = f"{g_roll}_{g_name}"
                            with open(FINGER_MAP_PATH, 'wb') as f:
                                pickle.dump(finger_map, f)
                            finger_status.success(f"✅ Slot {g_id} Enrolled!")
                            play_beep()
                            break
                    time.sleep(0.1)

# ── Mark Attendance ───────────────────────────────────────────────────────────
elif menu == "Mark Attendance":
    st.header("📷 Live Attendance Terminal")

    if ser and ser.in_waiting:
        line = ser.readline().decode(errors='ignore').strip()
        if "FOUND_ID:" in line:
            fid = line.split(":")[-1]
            if fid in finger_map:
                mark_attendance(finger_map[fid], "Fingerprint")
                st.toast(f"☝️ Fingerprint Match: {finger_map[fid]}", icon="✅")
                play_beep()

    webrtc_streamer(
        key="att",
        video_processor_factory=lambda: DualProcessor(mode="mark"),
        media_stream_constraints={"video": True, "audio": False}
    )

# ── View Records ──────────────────────────────────────────────────────────────
elif menu == "View Records":
    st.header("📊 Attendance Log")
    if os.path.exists(CSV_FILE):
        df = pd.read_csv(CSV_FILE, on_bad_lines='skip')
        st.dataframe(df.iloc[::-1], use_container_width=True)
        if st.button("🗑️ Clear Records"):
            os.remove(CSV_FILE)
            st.rerun()
    else:
        st.info("No records found.")