import streamlit as st
import cv2
import os
import pickle
import time
from face_utils import create_embedder, extract_face_crops, get_embedding
from train_model import retrain_system

if not os.path.exists('data'): os.makedirs('data')
st.set_page_config(page_title="Register Face", page_icon="👤")

st.title("👤 New Student Registration")
col1, col2 = st.columns(2)
name = col1.text_input("Name")
roll = col2.text_input("Roll No")

if st.button("Start Camera"):
    if not name or not roll:
        st.error("Enter Name and Roll Number!")
    else:
        status = st.empty()
        bar = st.progress(0)
        frame_window = st.image([])
        
        cap = cv2.VideoCapture(0)
        embedder = create_embedder()
        embeddings = []
        
        while len(embeddings) < 20:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            
            # Find face
            faces = extract_face_crops(frame)
            
            if faces:
                # Take the first face found for registration
                (x, y, w, h), face_crop = faces[0]
                
                # Get ID
                emb = get_embedding(face_crop, embedder)
                if emb is not None:
                    embeddings.append(emb)
                    # Dynamic Green Box
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            else:
                cv2.putText(frame, "No Face", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            bar.progress(len(embeddings) / 20)
            frame_window.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            time.sleep(0.05)
            
        cap.release()
        
        filename = f"data/{roll}_{name}.pkl"
        with open(filename, 'wb') as f:
            pickle.dump(embeddings, f)
            
        st.success("✅ Saved! Retraining model...")
        retrain_system()
        st.success("🚀 Done!")