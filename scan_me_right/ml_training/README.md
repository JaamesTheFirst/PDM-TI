# ML Training for Document Formatting Recognition

Scripts and resources for generating synthetic documents, training the TensorFlow Lite formatting classifier, and exporting artifacts consumed by the Flutter app.

## Overview

- **Input**: Synthetic A4 pages rendered at 1240x1754 plus cropped 224x224 blocks
- **Labels**: Multi-label flags per block (normal, bold, italic, underline, title, bullet list, numbered list)
- **Model**: MobileNetV3-Small backbone fine-tuned with sigmoid outputs (multi-label)
- **Output artifacts**:
  - `models/best_model.keras`
  - `models/formatting_classifier.tflite`
  - `models/training_history.png`

Current best model reaches ~80% aggregate accuracy. Lists and inline emphasis remain the weakest classes and are the focus of future fine-tuning.

## Quick Start

```bash
cd scan_me_right/ml_training
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python run_full_pipeline.py
```

The pipeline performs two steps:
1. `generate_training_data.py` creates synthetic pages, block crops, and metadata (stored in `training_data/`).
2. `train_model.py` loads the generated dataset, trains the classifier, exports Keras and TFLite models, and logs metrics.

## Data Generation Details

- **Page rendering**: Randomised layouts with titles, paragraphs, emphasis, bullet/numbered lists, underline, and optional dividers.
- **Fonts**: Mix of sans/serif fonts with size, spacing, and alignment jitter.
- **Augmentation**: Rotation, brightness/contrast jitter, Gaussian blur, and pixel noise.
- **Outputs**:
  - `training_data/pages/` full-page JPEGs
  - `training_data/images/` 224×224 crops
  - `training_data/labels/` JSON labels with multi-hot flags
  - `training_data/metadata.json` summary + per-sample metadata

Configuration constants (page count, probabilities, etc.) live at the top of `generate_training_data.py`.

## Training Overview

- Uses TensorFlow/Keras with MobileNetV3-Small base, global pooling, and sigmoid outputs.
- Loss: binary cross-entropy with optional class weighting.
- Metrics: binary accuracy and per-class precision/recall/AUC (reported to console).
- Split: 70% train / 15% validation / 15% test (configurable in `train_model.py`).
- TFLite export applies default optimizations for on-device inference.

To rerun training without regenerating data, skip step 1 above and call `python train_model.py` directly.

## Deployment to Flutter

1. Copy `models/formatting_classifier.tflite` to `assets/models/`.
2. Ensure `pubspec.yaml` lists `assets/models/formatting_classifier.tflite`.
3. Build/run the Flutter app; `FormattingModelService` loads the asset at startup.

## Future Improvements

- Generate richer mixed-format samples (bold+italic, nested lists, headers with subtitles).
- Incorporate public layout datasets (PubLayNet, DocBank) once download issues are resolved.
- Add automated threshold calibration and confusion-matrix logging to `train_model.py`.
- Experiment with focal loss or class weighting to lift list recall.

## Resources

- [TensorFlow Lite Guide](https://www.tensorflow.org/lite/guide)
- [MobileNetV3 Paper](https://arxiv.org/abs/1905.02244)
- [Flutter tflite_flutter plugin](https://pub.dev/packages/tflite_flutter)

