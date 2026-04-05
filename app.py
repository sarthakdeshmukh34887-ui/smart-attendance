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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

st.set_page_config(page_title="Biometric Attendance Dashboard", page_icon="🛡️", layout="wide")

def play_beep():
    st.components.v1.html("""
        <audio autoplay>
            <source src="https://cdn.pixabay.com/audio/2022/03/15/audio_507204595e.mp3" type="audio/mpeg">
        </audio>
    """, height=0)

# --- CONSTANTS ---
MODEL_PATH    = 'models/attendance_model.h5'
LABEL_PATH    = 'models/label_map.pkl'
RFID_MAP_PATH = 'models/rfid_map.pkl'
DATA_DIR      = 'data'
CSV_FILE      = "attendance.csv"
lock          = threading.Lock()

# ── Thresholds ────────────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.45
ENTROPY_THRESHOLD    = 1.50
COSINE_SIM_THRESHOLD = 0.20

# ── Face enhancement ──────────────────────────────────────────────────────────
def enhance_face(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

# ── Gate helpers ──────────────────────────────────────────────────────────────
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
    model, label_map, rfid_map, embedding_store = None, {}, {}, {}

    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
    if os.path.exists(LABEL_PATH):
        with open(LABEL_PATH, 'rb') as f:
            label_map = pickle.load(f)
    if os.path.exists(RFID_MAP_PATH):
        with open(RFID_MAP_PATH, 'rb') as f:
            rfid_map = pickle.load(f)

    for idx, name in label_map.items():
        pkl_path = os.path.join(DATA_DIR, f'{name}.pkl')
        if os.path.exists(pkl_path):
            with open(pkl_path, 'rb') as f:
                embedding_store[idx] = pickle.load(f)

    return embedder, model, label_map, rfid_map, embedding_store

embedder, model, label_map, rfid_map, embedding_store = load_resources()

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
# 3. VIDEO PROCESSOR
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

if ser: st.success(f"✅ RC522 RFID detected on **{selected_port}**")
else:   st.error("❌ Arduino not detected. Check COM Port.")

# ==========================================
# 5. NAVIGATION
# ==========================================
menu = st.sidebar.selectbox(
    "Navigation", ["Mark Attendance", "Register New User", "Manage RFID Cards", "View Records"]
)

# ── Register Face ─────────────────────────────────────────────────────────────
if menu == "Register New User":
    st.header("👤 Face Registration")
    st.info("💡 Register in the same lighting as attendance. Move head slightly for better accuracy.")
    c1, c2 = st.columns(2)
    f_name = c1.text_input("Name", key="fn")
    f_roll = c2.text_input("Roll No", key="fr")

    face_ctx = webrtc_streamer(
        key="face_reg",
        video_processor_factory=lambda: DualProcessor(mode="register"),
        media_stream_constraints={"video": True, "audio": False}
    )

    if st.button("💾 Save Face Samples"):
        if face_ctx.video_processor and len(face_ctx.video_processor.face_samples) >= 50:
            if not f_name or not f_roll:
                st.error("Enter Name and Roll No first!")
            else:
                label = f"{f_roll}_{f_name}"
                with open(f"data/{label}.pkl", 'wb') as f:
                    pickle.dump(face_ctx.video_processor.face_samples, f)
                with st.spinner("🔄 Retraining model..."):
                    success = retrain_system()
                if success:
                    st.cache_resource.clear()
                    st.success(f"✅ {f_name} registered!")
                    play_beep()
                    st.balloons()
                else:
                    st.error("❌ Need at least 2 registered students to train.")
        else:
            samples = len(face_ctx.video_processor.face_samples) if face_ctx.video_processor else 0
            st.warning(f"⚠️ Only {samples}/50 samples captured. Keep face in frame.")

# ── Manage RFID Cards ─────────────────────────────────────────────────────────
elif menu == "Manage RFID Cards":
    st.header("📡 RFID Card Management")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Register New Card")
        if not ser:
            st.warning("Connect Arduino first.")
        else:
            # Show registered students as dropdown
            if label_map:
                student_options = list(label_map.values())
                selected_student = st.selectbox("Select Student", student_options)
            else:
                st.warning("No students registered yet. Register faces first.")
                selected_student = None

            rfid_status = st.empty()

            if st.button("🔴 Scan Card to Register") and selected_student:
                rfid_status.info("📡 Waiting for RFID card... tap your card now")
                timeout = time.time() + 15
                registered = False
                while time.time() < timeout:
                    if ser.in_waiting:
                        line = ser.readline().decode(errors='ignore').strip()
                        if line.startswith("UID:"):
                            uid = line.split("UID:")[1].strip().upper()
                            rfid_map[uid] = selected_student
                            with open(RFID_MAP_PATH, 'wb') as f:
                                pickle.dump(rfid_map, f)
                            rfid_status.success(f"✅ Card {uid} linked to {selected_student}!")
                            play_beep()
                            registered = True
                            break
                    time.sleep(0.05)
                if not registered:
                    rfid_status.error("⏰ Timeout — no card detected. Try again.")

    with col2:
        st.subheader("Registered Cards")
        if rfid_map:
            rfid_df = pd.DataFrame([
                {"UID": uid, "Student": name}
                for uid, name in rfid_map.items()
            ])
            st.dataframe(rfid_df, use_container_width=True)

            # Delete a card
            uid_to_delete = st.selectbox("Select card to remove", list(rfid_map.keys()))
            if st.button("🗑️ Remove Card"):
                del rfid_map[uid_to_delete]
                with open(RFID_MAP_PATH, 'wb') as f:
                    pickle.dump(rfid_map, f)
                st.success(f"Card {uid_to_delete} removed.")
                st.rerun()
        else:
            st.info("No RFID cards registered yet.")

# ── Mark Attendance ───────────────────────────────────────────────────────────
elif menu == "Mark Attendance":
    st.header("📷 Live Attendance Terminal")

    # RFID listener
    if ser and ser.in_waiting:
        try:
            line = ser.readline().decode(errors='ignore').strip()
            if line.startswith("UID:"):
                uid = line.split("UID:")[1].strip().upper()
                if uid in rfid_map:
                    student = rfid_map[uid]
                    mark_attendance(student, "RFID")
                    st.toast(f"📡 RFID: {student.split('_')[-1]} marked present!", icon="✅")
                    play_beep()
                else:
                    st.toast(f"⚠️ Unknown card: {uid}", icon="❌")
        except:
            pass

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

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear All Records"):
                os.remove(CSV_FILE)
                st.rerun()
        with col2:
            st.download_button(
                "⬇️ Download CSV",
                data=df.to_csv(index=False),
                file_name=f"attendance_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    else:
        st.info("No records found.")