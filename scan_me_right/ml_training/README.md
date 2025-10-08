# ML Training for Document Formatting Recognition

This directory contains scripts and resources for training a custom TensorFlow Lite model to recognize text formatting in scanned documents.

## Overview

The goal is to train a lightweight CNN (MobileNet-based) model that can classify text regions into different formatting categories:
- **Normal text**
- **Bold text**
- **Italic text**
- **Underlined text**
- **Title/Header text**

## Quick Start

### 1. Install Dependencies

```bash
cd ml_training
pip install -r requirements.txt
```

### 2. Generate Training Data

```bash
python generate_training_data.py
```

This will create a `training_data/` directory with:
- `images/` - Synthetic document images
- `labels/` - JSON labels for each image
- `metadata.json` - Complete dataset metadata

### 3. Train the Model

(To be implemented in Phase 2)

```python
# Example training script structure
# train_model.py

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# Load data
# Build model
# Train
# Export to TFLite
```

### 4. Convert to TensorFlow Lite

```python
# Convert trained model to TFLite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Save
with open('formatting_classifier.tflite', 'wb') as f:
    f.write(tflite_model)
```

### 5. Deploy to Flutter App

1. Copy `.tflite` model to `assets/models/`
2. Update `pubspec.yaml` to include model as asset
3. Uncomment `tflite_flutter` dependency
4. Update `ocr_service.dart` to use ML model instead of heuristics

## Dataset Specifications

- **Image Size**: 800x600 pixels
- **Samples per Style**: 1,000 (configurable)
- **Total Samples**: 5,000 (5 styles × 1,000 samples)
- **Augmentations**:
  - Random rotation (-5° to +5°)
  - Brightness variation (0.8 to 1.2)
  - Contrast variation (0.9 to 1.1)
  - Gaussian blur (30% probability)
  - Random noise (20% probability)

## Model Architecture

Recommended architecture for offline mobile inference:

```
Input (224x224x3)
    ↓
MobileNetV3-Small (pretrained)
    ↓
GlobalAveragePooling2D
    ↓
Dense(128, activation='relu')
    ↓
Dropout(0.5)
    ↓
Dense(5, activation='softmax')
    ↓
Output (normal, bold, italic, underline, title)
```

## Training Tips

1. **Use Transfer Learning**: Start with pretrained MobileNetV3 weights
2. **Data Augmentation**: Apply real-world transformations (rotation, blur, lighting)
3. **Class Balancing**: Ensure equal samples per formatting style
4. **Validation Split**: 80% train, 10% validation, 10% test
5. **Early Stopping**: Monitor validation accuracy to prevent overfitting

## Performance Goals

- **Model Size**: < 10 MB
- **Inference Time**: < 100ms on mobile device
- **Accuracy**: > 90% on test set
- **Classes**: 5 formatting types

## Future Enhancements

- [ ] Add real document images from public datasets
- [ ] Support more formatting combinations (bold+italic, etc.)
- [ ] Detect font sizes more accurately
- [ ] Recognize text alignment (left, center, right)
- [ ] Support multiple languages

## Resources

- [TensorFlow Lite Guide](https://www.tensorflow.org/lite/guide)
- [MobileNet V3 Paper](https://arxiv.org/abs/1905.02244)
- [Flutter TFLite Plugin](https://pub.dev/packages/tflite_flutter)
- [DocBank Dataset](https://github.com/doc-analysis/DocBank)
- [RVL-CDIP Dataset](https://www.cs.cmu.edu/~aharley/rvl-cdip/)

## License

This training pipeline is part of the Scan Me Right project.

