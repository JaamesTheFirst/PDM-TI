"""
Training Script for Document Formatting Recognition Model
Optimized for CPU (16 cores) but will use GPU if available
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server environments
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.utils import Sequence

try:
    from generate_training_data import BLOCK_TYPES as DEFAULT_STYLE_ORDER
except ModuleNotFoundError:
    DEFAULT_STYLE_ORDER = []

# Configuration
DEFAULT_TRAINING_DATA_DIR = "training_data"
DEFAULT_MODEL_OUTPUT_DIR = "models"
STYLE_INDEX_FILENAME = "style_index.json"

TRAINING_DATA_DIR = DEFAULT_TRAINING_DATA_DIR
IMAGES_DIR = os.path.join(TRAINING_DATA_DIR, "images")
LABELS_DIR = os.path.join(TRAINING_DATA_DIR, "labels")
PAGES_DIR = os.path.join(TRAINING_DATA_DIR, "pages")
METADATA_FILE = os.path.join(TRAINING_DATA_DIR, "metadata.json")
MODEL_OUTPUT_DIR = DEFAULT_MODEL_OUTPUT_DIR
MODEL_CHECKPOINT_PATH = os.path.join(MODEL_OUTPUT_DIR, "best_model.keras")
TFLITE_MODEL_PATH = os.path.join(MODEL_OUTPUT_DIR, "formatting_classifier.tflite")

# Training hyperparameters (optimized for 16-core CPU)
BATCH_SIZE = 32  # Larger batch size for 16 cores
EPOCHS = 50
IMG_SIZE = (224, 224)  # MobileNetV3 input size
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.15
TEST_SPLIT = 0.15

def configure_paths(data_dir: str, model_dir: str) -> None:
    """Update global paths based on CLI arguments."""
    global TRAINING_DATA_DIR, IMAGES_DIR, LABELS_DIR, PAGES_DIR, METADATA_FILE
    global MODEL_OUTPUT_DIR, MODEL_CHECKPOINT_PATH, TFLITE_MODEL_PATH

    TRAINING_DATA_DIR = data_dir
    IMAGES_DIR = os.path.join(TRAINING_DATA_DIR, "images")
    LABELS_DIR = os.path.join(TRAINING_DATA_DIR, "labels")
    PAGES_DIR = os.path.join(TRAINING_DATA_DIR, "pages")
    METADATA_FILE = os.path.join(TRAINING_DATA_DIR, "metadata.json")

    MODEL_OUTPUT_DIR = model_dir
    MODEL_CHECKPOINT_PATH = os.path.join(MODEL_OUTPUT_DIR, "best_model.keras")
    TFLITE_MODEL_PATH = os.path.join(MODEL_OUTPUT_DIR, "formatting_classifier.tflite")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the document formatting recognition model.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=DEFAULT_TRAINING_DATA_DIR,
        help="Directory containing generated training_data (pages/images/labels/metadata).",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=DEFAULT_MODEL_OUTPUT_DIR,
        help="Directory where trained models and artifacts will be saved.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="Batch size for training.",
    )
    return parser.parse_args()


def configure_tensorflow_cpu():
    """Configure TensorFlow for optimal CPU performance"""
    print("Configuring TensorFlow for CPU optimization...")
    
    # Set thread count (use all 16 cores)
    tf.config.threading.set_intra_op_parallelism_threads(16)
    tf.config.threading.set_inter_op_parallelism_threads(16)
    
    # Enable optimizations
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'  # OneDNN optimizations
    os.environ['OMP_NUM_THREADS'] = '16'
    
    print("Configured for 16 CPU threads successfully")


def check_device():
    """Check if GPU is available"""
    print("=" * 60)
    print("Device Configuration")
    print("=" * 60)
    
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"GPU(s) found: {len(gpus)}")
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
        print("No GPU found. Training on CPU.")
        print(" Optimized for 16-core CPU performance.")
        configure_tensorflow_cpu()
        return False


class DataGenerator(Sequence):
    """Generator that loads images in batches to avoid OOM"""

    def __init__(self, samples, batch_size, img_size, shuffle=True):
        self.samples = samples
        self.batch_size = batch_size
        self.img_size = img_size
        self.shuffle = shuffle
        self.indices = np.arange(len(samples))
        if shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return int(np.ceil(len(self.samples) / self.batch_size))

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_samples = [self.samples[i] for i in batch_indices]

        images = []
        labels = []

        for sample in batch_samples:
            img_path = os.path.join(PAGES_DIR, sample['filename'])
            if os.path.exists(img_path):
                img = Image.open(img_path).convert('RGB')
                img = img.resize(self.img_size)
                img_array = np.array(img, dtype=np.float32) / 255.0
                images.append(img_array)

                labels.append(sample['label'])

        return np.array(images), np.array(labels, dtype=np.float32)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


def load_metadata() -> Tuple[Dict, float]:
    """Load metadata JSON and ensure required directories exist."""
    print("\n" + "=" * 60)
    print("Loading Dataset Metadata")
    print("=" * 60)

    if not os.path.exists(METADATA_FILE):
        raise FileNotFoundError(
            f"Metadata file not found: {METADATA_FILE}\n"
            "Please run generate_training_data.py first to create the dataset."
        )

    if not os.path.isdir(PAGES_DIR):
        raise FileNotFoundError(
            f"Page image directory not found: {PAGES_DIR}\n"
            "Please regenerate the dataset to include full-page renders."
        )

    start_time = time.time()
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    elapsed = time.time() - start_time

    if not metadata.get("pages"):
        raise ValueError("Metadata does not contain any page entries under the 'pages' key.")

    print(f"\n✅ Metadata loaded in {elapsed:.1f}s!")
    return metadata, elapsed


def derive_style_mapping(metadata: Dict) -> Dict[str, int]:
    """Infer the style taxonomy from metadata, preserving generator order when available."""
    style_set = set()
    for page in metadata.get("pages", []):
        style_set.update(page.get("feature_flags", {}).keys())
    if not style_set:
        style_set.update(
            block.get("style")
            for block in metadata.get("blocks", [])
            if isinstance(block, dict) and block.get("style")
        )

    style_set = {style for style in style_set if style}

    if not style_set:
        raise ValueError("Unable to determine style taxonomy from metadata.")

    ordered_styles: List[str] = []
    if DEFAULT_STYLE_ORDER:
        ordered_styles.extend([style for style in DEFAULT_STYLE_ORDER if style in style_set])

    remaining = sorted(style_set.difference(set(ordered_styles)))
    ordered_styles.extend(remaining)

    return {style: idx for idx, style in enumerate(ordered_styles)}


def persist_style_mapping(style_to_class: Dict[str, int]) -> str:
    """Persist style/index mapping for downstream consumers (e.g., mobile app)."""
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    ordered_styles = [style for style, _ in sorted(style_to_class.items(), key=lambda kv: kv[1])]
    mapping = {
        "styles": ordered_styles,
        "style_to_index": style_to_class,
    }
    mapping_path = os.path.join(MODEL_OUTPUT_DIR, STYLE_INDEX_FILENAME)
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2)
    return mapping_path


def load_dataset(metadata: Dict, style_to_class: Dict[str, int]):
    """Prepare train/val/test splits and generators."""
    pages = metadata.get("pages", [])
    samples = []
    class_totals = {style: 0 for style in style_to_class.keys()}

    for page in pages:
        feature_flags = page.get("feature_flags", {})
        label_vector = [0.0] * len(style_to_class)
        for style, class_idx in style_to_class.items():
            if feature_flags.get(style, 0):
                label_vector[class_idx] = 1.0
                class_totals[style] += 1
        samples.append(
            {
                "filename": page["filename"],
                "label": label_vector,
            }
        )

    print("\nPreparing dataset splits...")
    print(f"   Total pages: {len(samples):,}")
    print(f"   Classes: {len(style_to_class)}")

    print("\nClass distribution (pages containing feature):")
    for style, count in sorted(class_totals.items()):
        percentage = (count / len(samples) * 100) if samples else 0
        print(f"  {style:13s}: {count:4d} pages ({percentage:5.1f}%)")

    indices = np.arange(len(samples))
    val_test_fraction = VALIDATION_SPLIT + TEST_SPLIT
    train_indices, temp_indices = train_test_split(
        indices,
        test_size=val_test_fraction,
        random_state=42,
        shuffle=True,
    )
    if val_test_fraction == 0:
        val_indices = np.array([], dtype=int)
        test_indices = np.array([], dtype=int)
    else:
        relative_test = TEST_SPLIT / val_test_fraction
        val_indices, test_indices = train_test_split(
            temp_indices,
            test_size=relative_test,
            random_state=42,
            shuffle=True,
        )
        val_indices = np.array(val_indices)
        test_indices = np.array(test_indices)

    train_samples = [samples[i] for i in train_indices]
    val_samples = [samples[i] for i in val_indices]
    test_samples = [samples[i] for i in test_indices]

    print(f"\nDataset splits:")
    print(f"  Training:   {len(train_samples):,} pages ({len(train_samples)/len(samples)*100:.1f}%)")
    print(f"  Validation: {len(val_samples):,} pages ({len(val_samples)/len(samples)*100:.1f}%)")
    print(f"  Test:       {len(test_samples):,} pages ({len(test_samples)/len(samples)*100:.1f}%)")
    print(f"\n✅ Using batch loading - images will be loaded on-demand during training")
    print(f"   This avoids out-of-memory errors with large datasets.\n")

    train_gen = DataGenerator(train_samples, BATCH_SIZE, IMG_SIZE, shuffle=True)
    val_gen = DataGenerator(val_samples, BATCH_SIZE, IMG_SIZE, shuffle=False)
    test_gen = DataGenerator(test_samples, BATCH_SIZE, IMG_SIZE, shuffle=False)

    return train_gen, val_gen, test_gen, test_samples


def build_model(num_classes: int):
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
    print("Base model loaded successfully(frozen for transfer learning)")
    
    # Add custom classifier
    inputs = keras.Input(shape=(*IMG_SIZE, 3))
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='sigmoid')(x)
    
    model = Model(inputs, outputs)
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=[
            keras.metrics.BinaryAccuracy(name='binary_accuracy', threshold=0.5),
            keras.metrics.AUC(name='auc', multi_label=True, num_labels=num_classes),
        ]
    )
    
    print(f"Model built successfully!")
    print(f"Total parameters: {model.count_params():,}")
    trainable = sum([tf.size(w).numpy() for w in model.trainable_weights])
    print(f"Trainable parameters: {trainable:,}")
    print(f"Non-trainable parameters: {model.count_params() - trainable:,}")
    
    return model


def train_model(model, train_gen, val_gen):
    """Train the model using generators"""
    print("\n" + "=" * 60)
    print("Training Model")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_binary_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            MODEL_CHECKPOINT_PATH,
            monitor='val_binary_accuracy',
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
    print(f"  Training samples: {len(train_gen.samples):,}")
    print(f"  Validation samples: {len(val_gen.samples):,}")
    print(f"  Batches per epoch: {len(train_gen)}")
    print("\nStarting training...")
    print("(Using batch loading - images loaded on-demand to save memory)\n")
    
    start_time = time.time()
    
    # Train with generators
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1,
    )
    
    elapsed = time.time() - start_time
    print(f"\nTraining completed successfully in {elapsed/60:.1f} minutes ({elapsed:.0f} seconds)!")
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
    print(f"TFLite model saved: {TFLITE_MODEL_PATH}")
    print(f"   Model size: {file_size_mb:.2f} MB")
    
    if file_size_mb < 10:
        print(f"Model size goal achieved (< 10 MB)")
    else:
        print(f" Model size exceeds 10 MB goal")
    
    return TFLITE_MODEL_PATH


def plot_training_history(history):
    """Plot training history"""
    print("\n" + "=" * 60)
    print("Saving Training Plots")
    print("=" * 60)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot accuracy
    ax1.plot(history.history['binary_accuracy'], label='Training Binary Acc', marker='o')
    ax1.plot(history.history['val_binary_accuracy'], label='Validation Binary Acc', marker='s')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Binary Accuracy')
    ax1.set_title('Binary Accuracy (multi-label)')
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
    print(f"Training plots saved: {plot_path}")
    plt.close()


def evaluate_model(model, test_gen, test_samples, class_to_style: Dict[int, str]):
    """Evaluate model on test set"""
    print("\n" + "=" * 60)
    print("Evaluating Model")
    print("=" * 60)
    
    # Load best model if checkpoint exists
    if os.path.exists(MODEL_CHECKPOINT_PATH):
        print(f"Loading best model from checkpoint...")
        model = keras.models.load_model(MODEL_CHECKPOINT_PATH)
    
    # Evaluate using generator
    print("Evaluating on test set...")
    test_loss, test_binary_acc, test_auc = model.evaluate(test_gen, verbose=1)
 
    print(f"\nTest Results:")
    print(f"   Test Loss: {test_loss:.4f}")
    print(f"   Test Binary Accuracy: {test_binary_acc:.4f} ({test_binary_acc*100:.2f}%)")
    print(f"   Test AUC: {test_auc:.4f}")
    
    if test_binary_acc >= 0.90:
        print(f"Accuracy goal achieved (≥ 90%)")
    else:
        print(f"Accuracy below goal (≥ 90%)")
 
    # Get predictions for per-class metrics
    print("\nGenerating predictions for detailed metrics...")
    y_true = []
    y_pred = []

    for batch_idx in range(len(test_gen)):
        X_batch, y_batch = test_gen[batch_idx]
        pred_batch = model.predict(X_batch, verbose=0)
        y_true.append(y_batch)
        y_pred.append(pred_batch)

    y_true = np.vstack(y_true)
    y_pred = np.vstack(y_pred)
    y_pred_binary = (y_pred >= 0.5).astype(np.float32)

    print("\nPer-class metrics (threshold=0.5):")
    for class_id, style_name in sorted(class_to_style.items()):
        true_col = y_true[:, class_id]
        pred_col = y_pred_binary[:, class_id]
        support = int(true_col.sum())
        if support == 0:
            print(f"  {style_name:12s}: no positive examples in test set")
            continue

        tp = np.sum((true_col == 1) & (pred_col == 1))
        fp = np.sum((true_col == 0) & (pred_col == 1))
        fn = np.sum((true_col == 1) & (pred_col == 0))

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

        print(
            f"  {style_name:12s}: precision={precision:0.2f}, recall={recall:0.2f}, "
            f"F1={f1:0.2f}, support={support}"
        )

    return test_binary_acc


def main():
    """Main training pipeline"""
    args = parse_args()
    configure_paths(args.data_dir, args.model_dir)
    global EPOCHS, BATCH_SIZE
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size

    print("=" * 60)
    print("Document Formatting Recognition - Model Training")
    print("Optimized for 16-Core CPU")
    print("=" * 60)
    print(f"Data directory : {Path(TRAINING_DATA_DIR).resolve()}")
    print(f"Model directory: {Path(MODEL_OUTPUT_DIR).resolve()}")
    
    # Check device
    has_gpu = check_device()
    if not has_gpu:
        print("\nCPU Training Tips:")
        print("   - Using 16 CPU cores for parallel processing")
        print("   - Estimated training time: 2-5 hours")
        print("   - Monitor progress with callbacks (early stopping enabled)")
        print("   - Best model will be saved automatically")
    
    # Load dataset metadata and build style taxonomy
    metadata, _ = load_metadata()
    style_to_class = derive_style_mapping(metadata)
    class_to_style = {idx: style for style, idx in style_to_class.items()}
    mapping_path = persist_style_mapping(style_to_class)

    # Prepare generators
    train_gen, val_gen, test_gen, test_samples = load_dataset(metadata, style_to_class) 
    
    # Build model
    model = build_model(num_classes=len(style_to_class))
    
    # Train using generators
    history, model = train_model(model, train_gen, val_gen)
    
    # Evaluate using generator
    test_binary_acc = evaluate_model(model, test_gen, test_samples, class_to_style)
    
    # Plot history
    try:
        plot_training_history(history)
    except Exception as e:
        print(f"Could not generate plots: {e}")
        import traceback
        traceback.print_exc()
    
    # Convert to TFLite
    tflite_path = convert_to_tflite(model)
    
    # Final summary
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Best model saved: {MODEL_CHECKPOINT_PATH}")
    print(f"TFLite model saved: {tflite_path}")
    print(f"Style index saved: {mapping_path}")
    print(f"Test binary accuracy: {test_binary_acc*100:.2f}%")
    print("\nNext steps:")
    print("1. Review training plots: models/training_history.png")
    print("2. Copy the .tflite model to scan_me_right/assets/models/")
    print("3. Update pubspec.yaml to include the model")
    print("4. Integrate into the Flutter app")
    print("=" * 60)


if __name__ == "__main__":
    main()

