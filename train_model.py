import tensorflow as tf
import numpy as np
import os
import pickle
from tensorflow.keras.utils import to_categorical

def retrain_system():
    data_dir = 'data'
    model_dir = 'models'
    
    # Ensure directories exist
    if not os.path.exists(model_dir): 
        os.makedirs(model_dir)
    
    X = []
    y = []
    label_map = {} 
    current_label = 0
    
    # 1. Load Data from 'data/' folder
    files = [f for f in os.listdir(data_dir) if f.endswith('.pkl')]
    
    if not files:
        print("⚠️ No student data found! Please register someone first.")
        return False

    print(f"🔄 Loading data for {len(files)} students...")
    
    for filename in files:
        path = os.path.join(data_dir, filename)
        with open(path, 'rb') as f:
            embeddings = pickle.load(f)
            
        for emb in embeddings:
            X.append(emb)
            y.append(current_label)
        
        # Map ID to Name (e.g., 0 -> "23_Om")
        label_map[current_label] = filename.replace('.pkl', '')
        current_label += 1

    X = np.array(X)
    y = np.array(y)
    
    # One-hot encode labels
    y_cat = to_categorical(y, num_classes=current_label)

    # --- CRITICAL FIX: Get the input shape dynamically ---
    # This detects if your embeddings are 128 or 1024 size
    input_shape = X.shape[1] 
    print(f"🧠 Detected Embedding Size: {input_shape}")

    # 2. Define the Neural Network
    model = tf.keras.Sequential([
        # Use the detected shape instead of hardcoding 128
        tf.keras.layers.Input(shape=(input_shape,)), 
        tf.keras.layers.Dense(512, activation='relu'), # Increased for better 1024 support
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(current_label, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # 3. Train
    print("🚀 Training Neural Network...")
    model.fit(X, y_cat, epochs=15, batch_size=4, verbose=0)
    
    # 4. Save
    model.save(os.path.join(model_dir, 'attendance_model.h5'))
    with open(os.path.join(model_dir, 'label_map.pkl'), 'wb') as f:
        pickle.dump(label_map, f)
        
    print("✅ System Retrained Successfully!")
    return True

if __name__ == "__main__":
    retrain_system()