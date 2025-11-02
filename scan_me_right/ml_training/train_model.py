"""
Training Script for Document Formatting Recognition Model
Optimized for CPU (16 cores) but will use GPU if available
"""

import os
import json
import numpy as np
from pathlib import Path
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments
import matplotlib.pyplot as plt
import time

# Configuration
TRAINING_DATA_DIR = "training_data"
IMAGES_DIR = os.path.join(TRAINING_DATA_DIR, "images")
LABELS_DIR = os.path.join(TRAINING_DATA_DIR, "labels")
METADATA_FILE = os.path.join(TRAINING_DATA_DIR, "metadata.json")
MODEL_OUTPUT_DIR = "models"
MODEL_CHECKPOINT_PATH = os.path.join(MODEL_OUTPUT_DIR, "best_model.keras")
TFLITE_MODEL_PATH = os.path.join(MODEL_OUTPUT_DIR, "formatting_classifier.tflite")

# Training hyperparameters (optimized for 16-core CPU)
BATCH_SIZE = 32  # Larger batch size for 16 cores
EPOCHS = 50
IMG_SIZE = (224, 224)  # MobileNetV3 input size
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.1
TEST_SPLIT = 0.1

# Style mapping
STYLE_TO_CLASS = {
    'normal': 0,
    'bold': 1,
    'italic': 2,
    'underline': 3,
    'title': 4
}

CLASS_TO_STYLE = {v: k for k, v in STYLE_TO_CLASS.items()}


def configure_tensorflow_cpu():
    """Configure TensorFlow for optimal CPU performance"""
    print("Configuring TensorFlow for CPU optimization...")
    
    # Set thread count (use all 16 cores)
    tf.config.threading.set_intra_op_parallelism_threads(16)
    tf.config.threading.set_inter_op_parallelism_threads(16)
    
    # Enable optimizations
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'  # OneDNN optimizations
    os.environ['OMP_NUM_THREADS'] = '16'
    
    print("  ✅ Configured for 16 CPU threads")


def check_device():
    """Check if GPU is available"""
    print("=" * 60)
    print("Device Configuration")
    print("=" * 60)
    
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✅ GPU(s) found: {len(gpus)}")
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu.name}")
        try:
            # Enable memory growth to avoid allocating all GPU memory
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"  Warning: {e}")
        return True
    else:
        print("⚠️  No GPU found. Training on CPU.")
        print("   Optimized for 16-core CPU performance.")
        configure_tensorflow_cpu()
        return False


def load_dataset():
    """Load images and labels from the training data directory"""
    print("\n" + "=" * 60)
    print("Loading Dataset")
    print("=" * 60)
    
    if not os.path.exists(METADATA_FILE):
        raise FileNotFoundError(
            f"Metadata file not found: {METADATA_FILE}\n"
            "Please run generate_training_data.py first to create the dataset."
        )
    
    # Load metadata
    with open(METADATA_FILE, 'r') as f:
        metadata = json.load(f)
    
    images = []
    labels = []
    
    print(f"Loading {len(metadata['samples'])} samples...")
    start_time = time.time()
    
    for i, sample in enumerate(metadata['samples']):
        if (i + 1) % 500 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            remaining = (len(metadata['samples']) - i - 1) / rate
            print(f"  Loaded {i + 1}/{len(metadata['samples'])} samples "
                  f"({rate:.1f} samples/sec, ~{remaining:.0f}s remaining)")
        
        # Load image
        img_path = os.path.join(IMAGES_DIR, sample['filename'])
        if not os.path.exists(img_path):
            print(f"  Warning: Image not found: {img_path}")
            continue
        
        img = Image.open(img_path).convert('RGB')
        img = img.resize(IMG_SIZE)
        img_array = np.array(img) / 255.0  # Normalize to [0, 1]
        images.append(img_array)
        
        # Get label
        style = sample['style']
        labels.append(STYLE_TO_CLASS.get(style, 0))
    
    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)
    
    elapsed = time.time() - start_time
    print(f"\n✅ Dataset loaded successfully in {elapsed:.1f}s!")
    print(f"   Images shape: {images.shape}")
    print(f"   Labels shape: {labels.shape}")
    print(f"   Classes: {len(STYLE_TO_CLASS)}")
    
    # Print class distribution
    unique, counts = np.unique(labels, return_counts=True)
    print("\nClass distribution:")
    for label, count in zip(unique, counts):
        print(f"  {CLASS_TO_STYLE[label]}: {count} samples")
    
    return images, labels


def build_model(num_classes=5):
    """Build MobileNetV3-based model with transfer learning"""
    print("\n" + "=" * 60)
    print("Building Model")
    print("=" * 60)
    
    # Load MobileNetV3-Small with pretrained ImageNet weights
    base_model = MobileNetV3Small(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights='imagenet',
        alpha=1.0,
        minimalistic=False
    )
    
    # Freeze base model initially (transfer learning - only train classifier head)
    base_model.trainable = False
    print("✅ Base model loaded (frozen for transfer learning)")
    
    # Add custom classifier
    inputs = keras.Input(shape=(*IMG_SIZE, 3))
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"✅ Model built successfully!")
    print(f"   Total parameters: {model.count_params():,}")
    trainable = sum([tf.size(w).numpy() for w in model.trainable_weights])
    print(f"   Trainable parameters: {trainable:,}")
    print(f"   Non-trainable parameters: {model.count_params() - trainable:,}")
    
    return model


def train_model(model, X_train, y_train, X_val, y_val):
    """Train the model"""
    print("\n" + "=" * 60)
    print("Training Model")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            MODEL_CHECKPOINT_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    print(f"Training configuration:")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Validation samples: {len(X_val)}")
    print("\nStarting training...\n")
    
    start_time = time.time()
    
    # Train
    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    elapsed = time.time() - start_time
    print(f"\n✅ Training completed in {elapsed/60:.1f} minutes ({elapsed:.0f} seconds)!")
    return history, model


def convert_to_tflite(model):
    """Convert trained model to TensorFlow Lite format"""
    print("\n" + "=" * 60)
    print("Converting to TensorFlow Lite")
    print("=" * 60)
    
    # Load best model if checkpoint exists
    if os.path.exists(MODEL_CHECKPOINT_PATH):
        print(f"Loading best model from checkpoint: {MODEL_CHECKPOINT_PATH}")
        model = keras.models.load_model(MODEL_CHECKPOINT_PATH)
    
    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Convert
    print("Converting model...")
    tflite_model = converter.convert()
    
    # Save
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    with open(TFLITE_MODEL_PATH, 'wb') as f:
        f.write(tflite_model)
    
    file_size_mb = os.path.getsize(TFLITE_MODEL_PATH) / (1024 * 1024)
    print(f"✅ TFLite model saved: {TFLITE_MODEL_PATH}")
    print(f"   Model size: {file_size_mb:.2f} MB")
    
    if file_size_mb < 10:
        print(f"   ✅ Model size goal achieved (< 10 MB)")
    else:
        print(f"   ⚠️  Model size exceeds 10 MB goal")
    
    return TFLITE_MODEL_PATH


def plot_training_history(history):
    """Plot training history"""
    print("\n" + "=" * 60)
    print("Saving Training Plots")
    print("=" * 60)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy', marker='o')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='s')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.legend()
    ax1.grid(True)
    ax1.set_ylim([0, 1])
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss', marker='o')
    ax2.plot(history.history['val_loss'], label='Validation Loss', marker='s')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Model Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plot_path = os.path.join(MODEL_OUTPUT_DIR, 'training_history.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✅ Training plots saved: {plot_path}")
    plt.close()


def evaluate_model(model, X_test, y_test):
    """Evaluate model on test set"""
    print("\n" + "=" * 60)
    print("Evaluating Model")
    print("=" * 60)
    
    # Load best model if checkpoint exists
    if os.path.exists(MODEL_CHECKPOINT_PATH):
        print(f"Loading best model from checkpoint...")
        model = keras.models.load_model(MODEL_CHECKPOINT_PATH)
    
    # Evaluate
    print("Evaluating on test set...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1, batch_size=BATCH_SIZE)
    
    print(f"\n✅ Test Results:")
    print(f"   Test Loss: {test_loss:.4f}")
    print(f"   Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    if test_accuracy >= 0.90:
        print(f"   ✅ Accuracy goal achieved (≥ 90%)")
    else:
        print(f"   ⚠️  Accuracy below goal (≥ 90%)")
    
    # Get predictions
    print("\nGenerating predictions...")
    y_pred = model.predict(X_test, batch_size=BATCH_SIZE, verbose=1)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Print per-class accuracy
    print("\nPer-class accuracy:")
    for class_id, style_name in CLASS_TO_STYLE.items():
        mask = y_test == class_id
        if np.sum(mask) > 0:
            class_acc = np.mean(y_pred_classes[mask] == y_test[mask])
            count = np.sum(mask)
            print(f"  {style_name:12s}: {class_acc*100:5.2f}% ({count} samples)")
    
    return test_accuracy


def main():
    """Main training pipeline"""
    print("=" * 60)
    print("Document Formatting Recognition - Model Training")
    print("Optimized for 16-Core CPU")
    print("=" * 60)
    
    # Check device
    has_gpu = check_device()
    if not has_gpu:
        print("\n💡 CPU Training Tips:")
        print("   - Using 16 CPU cores for parallel processing")
        print("   - Estimated training time: 2-5 hours")
        print("   - Monitor progress with callbacks (early stopping enabled)")
        print("   - Best model will be saved automatically")
    
    # Load dataset
    images, labels = load_dataset()
    
    # Split dataset
    print("\n" + "=" * 60)
    print("Splitting Dataset")
    print("=" * 60)
    X_temp, X_test, y_temp, y_test = train_test_split(
        images, labels, test_size=TEST_SPLIT, random_state=42, stratify=labels
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=VALIDATION_SPLIT/(1-TEST_SPLIT), 
        random_state=42, stratify=y_temp
    )
    
    print(f"Dataset splits:")
    print(f"  Training:   {len(X_train):5d} samples ({len(X_train)/len(images)*100:.1f}%)")
    print(f"  Validation: {len(X_val):5d} samples ({len(X_val)/len(images)*100:.1f}%)")
    print(f"  Test:       {len(X_test):5d} samples ({len(X_test)/len(images)*100:.1f}%)")
    
    # Build model
    model = build_model(num_classes=len(STYLE_TO_CLASS))
    
    # Train
    history, model = train_model(model, X_train, y_train, X_val, y_val)
    
    # Evaluate
    test_accuracy = evaluate_model(model, X_test, y_test)
    
    # Plot history
    try:
        plot_training_history(history)
    except Exception as e:
        print(f"⚠️  Could not generate plots: {e}")
        import traceback
        traceback.print_exc()
    
    # Convert to TFLite
    tflite_path = convert_to_tflite(model)
    
    # Final summary
    print("\n" + "=" * 60)
    print("Training Complete! 🎉")
    print("=" * 60)
    print(f"✅ Best model saved: {MODEL_CHECKPOINT_PATH}")
    print(f"✅ TFLite model saved: {tflite_path}")
    print(f"✅ Test accuracy: {test_accuracy*100:.2f}%")
    print("\nNext steps:")
    print("1. Review training plots: models/training_history.png")
    print("2. Copy the .tflite model to scan_me_right/assets/models/")
    print("3. Update pubspec.yaml to include the model")
    print("4. Integrate into the Flutter app")
    print("=" * 60)


if __name__ == "__main__":
    main()

