import cv2
import numpy as np
import tensorflow as tf
import pickle
import pandas as pd
import os
from datetime import datetime
from scipy.spatial.distance import cosine
from face_utils import create_embedder, extract_face_crops, get_embedding

# Config
MODEL_PATH = os.path.join('models', 'attendance_model.h5')
LABEL_PATH = os.path.join('models', 'label_map.pkl')
DATA_DIR   = 'data'
CSV_FILE   = "attendance.csv"

if not os.path.exists(MODEL_PATH):
    print("❌ Model not found! Run 'register.py' first.")
    exit()

# ── Load AI ───────────────────────────────────────────────────────────────────
print("⏳ Loading AI Model...")
model = tf.keras.models.load_model(MODEL_PATH)
with open(LABEL_PATH, 'rb') as f:
    label_map = pickle.load(f)

# Load stored embeddings for cosine similarity check
print("⏳ Loading stored embeddings...")
embedding_store = {}  # {class_id: [emb, emb, ...]}
for idx, name in label_map.items():
    pkl_path = os.path.join(DATA_DIR, f'{name}.pkl')
    if os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as f:
            embedding_store[idx] = pickle.load(f)

embedder = create_embedder()
session_attendance = set()

# ── Thresholds — raise CONFIDENCE or lower COSINE if unknowns still slip through
CONFIDENCE_THRESHOLD = 0.60   # was 0.80
ENTROPY_THRESHOLD    = 0.90   # was 0.65
COSINE_SIM_THRESHOLD = 0.40   # was 0.60

# ── Helper functions ──────────────────────────────────────────────────────────
def softmax_entropy(probs: np.ndarray) -> float:
    """High entropy = model is unsure = likely unknown face."""
    probs = np.clip(probs, 1e-9, 1.0)
    return float(-np.sum(probs * np.log(probs)))

def best_cosine_sim(embedding: np.ndarray, class_id: int) -> float:
    """Compare face embedding directly against every stored sample for that class."""
    stored = embedding_store.get(class_id, [])
    if not stored:
        return 0.0
    return max(1 - cosine(embedding, s) for s in stored)

def is_known_face(emb: np.ndarray, probs: np.ndarray, class_id: int) -> tuple:
    """
    Returns (is_known: bool, confidence: float, entropy: float, cos_sim: float)
    ALL three gates must pass, otherwise → Unknown.
    """
    confidence = float(np.max(probs))
    entropy    = softmax_entropy(probs)
    cos_sim    = best_cosine_sim(emb, class_id)

    known = (
        confidence >= CONFIDENCE_THRESHOLD and
        entropy    <= ENTROPY_THRESHOLD    and
        cos_sim    >= COSINE_SIM_THRESHOLD
    )
    return known, confidence, entropy, cos_sim

# ── Attendance marking (unchanged) ───────────────────────────────────────────
def mark_attendance(student_label):
    try:
        roll, name = student_label.split('_', 1)
    except ValueError:
        roll, name = "Unknown", student_label

    if student_label not in session_attendance:
        now = datetime.now()
        data = {
            "Date":   [now.strftime("%Y-%m-%d")],
            "Time":   [now.strftime("%H:%M:%S")],
            "Roll":   [roll],
            "Name":   [name],
            "Status": ["Present"],
        }
        df = pd.DataFrame(data)
        if not os.path.exists(CSV_FILE):
            df.to_csv(CSV_FILE, index=False)
        else:
            df.to_csv(CSV_FILE, mode='a', header=False, index=False)

        session_attendance.add(student_label)
        print(f"✅ MARKED: {name}")

# ── Camera loop ───────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces_found = extract_face_crops(frame)

    for (x, y, w, h), face_crop in faces_found:
        emb = get_embedding(face_crop, embedder)

        if emb is not None:
            # L2-normalize — must match how training embeddings were normalized
            emb = emb / max(np.linalg.norm(emb), 1e-8)

            probs    = model.predict(np.expand_dims(emb, axis=0), verbose=0)[0]
            class_id = int(np.argmax(probs))

            known, conf, entropy, cos_sim = is_known_face(emb, probs, class_id)

            if known:
                student_label = label_map[class_id]
                color = (0, 255, 0)  # green
                try:
                    display_name = student_label.split('_', 1)[1]
                except:
                    display_name = student_label
                text = f"{display_name} {int(conf*100)}%"
                mark_attendance(student_label)
            else:
                color = (0, 0, 255)  # red
                text  = "Unknown"
                # Uncomment to debug why a face is being rejected:
                print(f"[REJECTED] conf={conf:.2f} entropy={entropy:.3f} cos={cos_sim:.3f}")

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, text, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("Smart Attendance", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()