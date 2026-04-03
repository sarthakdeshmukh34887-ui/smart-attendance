import tensorflow as tf
import numpy as np
import os
import pickle
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight


def retrain_system():
    data_dir  = 'data'
    model_dir = 'models'

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    X, y = [], []
    label_map     = {}
    current_label = 0

    # 1. Load data
    files = [f for f in os.listdir(data_dir) if f.endswith('.pkl')]
    if not files:
        print("⚠️  No student data found! Please register someone first.")
        return False

    print(f"🔄 Loading data for {len(files)} students...")

    for filename in files:
        path = os.path.join(data_dir, filename)
        with open(path, 'rb') as f:
            embeddings = pickle.load(f)

        for emb in embeddings:
            X.append(emb)
            y.append(current_label)

        label_map[current_label] = filename.replace('.pkl', '')
        current_label += 1

    X = np.array(X, dtype=np.float32)
    y = np.array(y)

    # 2. L2 normalize — must match normalization done in main_attenance.py
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X = X / np.maximum(norms, 1e-8)

    input_shape = X.shape[1]
    print(f"🧠 Detected embedding size: {input_shape}, classes: {current_label}")

    # 3. Guard — need at least 2 students for a meaningful classifier
    if current_label < 2:
        print("⚠️  Only 1 student registered.")
        print("   Register at least one more person, then retrain.")
        return False

    # 4. One-hot encode labels
    y_cat = to_categorical(y, num_classes=current_label)

    # 5. Stratified train/val split
    #    stratify=y ensures every class appears in both train and val
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_cat,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # 6. Class weights — handles unequal samples per student
    raw_labels    = np.argmax(y_train, axis=1)
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(raw_labels),
        y=raw_labels
    )
    class_weight_dict = dict(enumerate(class_weights))

    # 7. Model — BatchNorm + Dropout + L2 regularization
    reg = tf.keras.regularizers.l2(1e-4)

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_shape,)),

        tf.keras.layers.Dense(512, kernel_regularizer=reg),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dropout(0.4),

        tf.keras.layers.Dense(256, kernel_regularizer=reg),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(128, kernel_regularizer=reg),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Dense(current_label, activation='softmax'),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # 8. Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(model_dir, 'best_model.keras'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=0
        ),
    ]

    # 9. Train
    batch_size = min(32, max(8, len(X_train) // 10))
    print(f"🚀 Training with batch_size={batch_size}, up to 50 epochs...")

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=batch_size,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1
    )

    # 10. Report
    val_acc = max(history.history['val_accuracy'])
    print(f"✅ Best validation accuracy: {val_acc:.2%}")

    # 11. Save model + label map
    model.save(os.path.join(model_dir, 'attendance_model.h5'))
    with open(os.path.join(model_dir, 'label_map.pkl'), 'wb') as f:
        pickle.dump(label_map, f)

    print("✅ System retrained successfully!")
    return True


if __name__ == "__main__":
    retrain_system()