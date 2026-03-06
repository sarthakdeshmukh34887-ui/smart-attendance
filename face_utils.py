import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import numpy as np
import os

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
detector = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

def create_embedder():
    # 1. Define path
    model_path = os.path.join('models', 'face_embedder.tflite')
    
    # 2. Check if file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Error: Model file not found at: {os.path.abspath(model_path)}")

    # 3. FIX: Read the file into memory (RAM) instead of passing the path
    # This bypasses the Windows path glitch entirely.
    with open(model_path, 'rb') as f:
        model_data = f.read()

    base_options = python.BaseOptions(model_asset_buffer=model_data)
    
    options = vision.ImageEmbedderOptions(
        base_options=base_options,
        l2_normalize=True, 
        quantize=False
    )
    return vision.ImageEmbedder.create_from_options(options)

def extract_face_crops(image):
    """
    Finds all faces in an image.
    Returns a list of tuples: (bounding_box_rect, cropped_face_image)
    """
    if image is None: return []
    
    height, width, _ = image.shape
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = detector.process(image_rgb)
    
    faces = []
    
    if results.detections:
        for detection in results.detections:
            # 1. Get Dynamic Bounding Box
            bboxC = detection.location_data.relative_bounding_box
            x = int(bboxC.xmin * width)
            y = int(bboxC.ymin * height)
            w = int(bboxC.width * width)
            h = int(bboxC.height * height)
            
            # Ensure box is within image checking boundaries
            x, y = max(0, x), max(0, y)
            w, h = min(w, width - x), min(h, height - y)
            
            # 2. Crop the face
            face_crop = image[y:y+h, x:x+w]
            
            if face_crop.size > 0:
                faces.append(((x, y, w, h), face_crop))
                
    return faces

def get_embedding(face_crop, embedder):
    """
    Takes a CROPPED face image and returns the embedding vector.
    """
    if face_crop is None or face_crop.size == 0:
        return None
        
    # Convert to MediaPipe Image format
    image_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    
    embedding_result = embedder.embed(mp_image)
    
    if embedding_result.embeddings:
        return embedding_result.embeddings[0].embedding
    return None