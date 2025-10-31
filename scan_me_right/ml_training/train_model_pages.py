"""
Train formatting classifier using full document pages
Each page contains multiple text blocks with different styles
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from PIL import Image
from sklearn.model_selection import train_test_split
from datetime import datetime

print("TensorFlow version:", tf.__version__)
print("GPU Available:", tf.config.list_physical_devices('GPU'))

# Configuration
TRAINING_DATA_DIR = "training_data_comprehensive"
IMAGES_DIR = os.path.join(TRAINING_DATA_DIR, "images")
LABELS_DIR = os.path.join(TRAINING_DATA_DIR, "labels")
MODEL_OUTPUT = "formatting_classifier.h5"
TFLITE_OUTPUT = "formatting_classifier.tflite"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 25
LEARNING_RATE = 0.001

# All formatting classes
CLASSES = [
    'normal',           # 0 - Regular paragraph text
    'bold',            # 1 - Bold text
    'italic',          # 2 - Italic text
    'underline',       # 3 - Underlined text
    'bold_italic',     # 4 - Bold + Italic
    'title_h1',        # 5 - Main title/header 1
    'header_h2',       # 6 - Section header 2
    'header_h3',       # 7 - Subsection header 3
    'bullet_point',    # 8 - Bullet list item
    'numbered_list',   # 9 - Numbered list item
    'blockquote',      # 10 - Indented quote
    'horizontal_line', # 11 - Divider line
    'indented_paragraph', # 12 - Indented text
    'inline_normal',   # 13 - Normal text in mixed line
    'inline_bold',     # 14 - Bold text in mixed line
    'inline_italic',   # 15 - Italic text in mixed line
]
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(CLASSES)}


def load_dataset():
    """Load dataset from full document pages"""
    print("\n Loading dataset from document pages...")
    
    images = []
    labels = []
    
    label_files = [f for f in os.listdir(LABELS_DIR) if f.endswith('.json')]
    total_blocks = 0
    
    for i, label_file in enumerate(label_files):
        if (i + 1) % 100 == 0:
            print(f"  Processing document {i + 1}/{len(label_files)}... (Total blocks: {total_blocks})")
        
        with open(os.path.join(LABELS_DIR, label_file), 'r') as f:
            doc_data = json.load(f)
        
        image_path = os.path.join(IMAGES_DIR, doc_data['filename'])
        
        if not os.path.exists(image_path):
            continue
        
        try:
            # Load full page image
            full_image = Image.open(image_path).convert('RGB')
            
            # Extract each text block from the page
            for block in doc_data['text_blocks']:
                # Crop text region
                x, y = int(block['x']), int(block['y'])
                w, h = int(block['width']), int(block['height'])
                
                # Add padding
                padding = 10
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = min(full_image.width - x, w + 2*padding)
                h = min(full_image.height - y, h + 2*padding)
                
                # Crop and resize
                cropped = full_image.crop((x, y, x + w, y + h))
                resized = cropped.resize(IMG_SIZE)
                img_array = np.array(resized) / 255.0
                
                images.append(img_array)
                labels.append(CLASS_TO_IDX[block['style']])
                total_blocks += 1
                
        except Exception as e:
            print(f"  Error processing {image_path}: {e}")
    
    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)
    
    print(f"\n  Loaded {len(images):,} text blocks from {len(label_files):,} documents")
    print(f"  Image shape: {images.shape}")
    print(f"  Labels shape: {labels.shape}")
    
    # Print class distribution
    print("\n Class distribution:")
    unique, counts = np.unique(labels, return_counts=True)
    for cls_idx, count in zip(unique, counts):
        print(f"  {CLASSES[cls_idx]}: {count:,} samples ({count/len(labels)*100:.1f}%)")
    
    return images, labels


def create_model():
    """Create MobileNetV3 model"""
    print("\n Building model...")
    
    base_model = keras.applications.MobileNetV3Small(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights='imagenet',
        pooling='avg'
    )
    
    # Freeze base model
    base_model.trainable = False
    
    # Larger network for 16 classes
    model = keras.Sequential([
        base_model,
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(len(CLASSES), activation='softmax')  # 16 output classes
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("  Model created")
    model.summary()
    
    return model


def train_model(model, X_train, y_train, X_val, y_val):
    """Train the model"""
    print("\n Training model...")
    print(f"  Training samples: {len(X_train):,}")
    print(f"  Validation samples: {len(X_val):,}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print()
    
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            'best_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    return history


def evaluate_model(model, X_test, y_test):
    """Evaluate model"""
    print("\n Evaluating model...")
    
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"  Test Loss: {loss:.4f}")
    print(f"  Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    predictions = model.predict(X_test, verbose=0)
    pred_classes = np.argmax(predictions, axis=1)
    
    print("\n Per-class accuracy:")
    for i, class_name in enumerate(CLASSES):
        class_mask = y_test == i
        if class_mask.sum() > 0:
            class_acc = (pred_classes[class_mask] == y_test[class_mask]).mean()
            class_count = class_mask.sum()
            print(f"  {class_name:12s}: {class_acc:.4f} ({class_acc*100:.2f}%) - {class_count:,} samples")
    
    return accuracy


def convert_to_tflite(model):
    """Convert to TensorFlow Lite"""
    print("\n Converting to TensorFlow Lite...")
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    
    tflite_model = converter.convert()
    
    with open(TFLITE_OUTPUT, 'wb') as f:
        f.write(tflite_model)
    
    size_mb = os.path.getsize(TFLITE_OUTPUT) / (1024 * 1024)
    print(f"  TFLite model saved: {TFLITE_OUTPUT}")
    print(f"  Model size: {size_mb:.2f} MB")
    
    return tflite_model


def test_tflite_model(tflite_model, X_test, y_test, num_samples=100):
    """Test TFLite inference"""
    print("\n Testing TFLite model...")
    
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    correct = 0
    for i in range(min(num_samples, len(X_test))):
        input_data = np.expand_dims(X_test[i], axis=0).astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        pred_class = np.argmax(output[0])
        
        if pred_class == y_test[i]:
            correct += 1
    
    accuracy = correct / min(num_samples, len(X_test))
    print(f"  TFLite Accuracy ({min(num_samples, len(X_test))} samples): {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    return accuracy


def main():
    print("=" * 70)
    print("Document Formatting Classifier Training (Full Pages)")
    print("=" * 70)
    
    if not os.path.exists(IMAGES_DIR):
        print("\n Error: Training data not found!")
        print(f"   Please run 'python3 generate_document_pages.py' first")
        return
    
    # Load dataset
    X, y = load_dataset()
    
    if len(X) == 0:
        print("\n Error: No training data loaded!")
        return
    
    # Split: 80% train, 10% val, 10% test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"\n Dataset split:")
    print(f"  Training:   {len(X_train):,} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"  Validation: {len(X_val):,} samples ({len(X_val)/len(X)*100:.1f}%)")
    print(f"  Test:       {len(X_test):,} samples ({len(X_test)/len(X)*100:.1f}%)")
    
    # Create and train
    model = create_model()
    history = train_model(model, X_train, y_train, X_val, y_val)
    
    # Evaluate
    test_accuracy = evaluate_model(model, X_test, y_test)
    
    # Save
    model.save(MODEL_OUTPUT)
    print(f"\n  Keras model saved: {MODEL_OUTPUT}")
    
    # Convert
    tflite_model = convert_to_tflite(model)
    test_tflite_model(tflite_model, X_test, y_test)
    
    # Summary
    print("\n" + "=" * 70)
    print("✅ Training Complete!")
    print("=" * 70)
    print(f"  Final Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"  Models saved:")
    print(f"    - Keras:  {MODEL_OUTPUT}")
    print(f"    - TFLite: {TFLITE_OUTPUT}")
    print()
    print("  Next steps:")
    print(f"    1. cp {TFLITE_OUTPUT} ../assets/models/")
    print(f"    2. Update Flutter app to use the model")
    print("=" * 70)


if __name__ == "__main__":
    main()

