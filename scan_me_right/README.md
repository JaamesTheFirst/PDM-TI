# Scan Me Right

Privacy-first offline document scanner with OCR, local persistence, PDF/TXT export, and an on-device TensorFlow Lite model that predicts formatting.

## Project Goal

Provide a secure alternative to cloud-backed scanners. Every step - capture, OCR, formatting recognition, storage, and export - runs entirely on-device.

## Feature Status

### Currently Implemented
- Camera capture with optional gallery import
- Offline OCR powered by Google ML Kit
- Local document database (SQLite)
- Export to PDF (styled) and TXT (plain)
- On-device formatting classifier (TensorFlow Lite) integrated into the OCR pipeline
- Share/export documents to other applications

### In Progress
- Improve formatting accuracy (lists and emphasis currently underperform relative to paragraphs/titles)
- DOCX export
- Batch scanning workflow
- Optional encrypted sync (deferred)

## Tech Stack

- Flutter 3.35 / Dart 3.9
- State management: provider
- OCR: google_mlkit_text_recognition
- Persistence: sqflite + path_provider
- Export: pdf, printing
- ML runtime: tflite_flutter
- Synthetic data & training: TensorFlow/Keras + Pillow/NumPy in `ml_training/`

## Project Structure

```
scan_me_right/
|-- lib/
|   |-- config/
|   |-- models/
|   |-- providers/
|   |-- screens/
|   |-- services/
|   |-- utils/
|   |-- widgets/
|   `-- main.dart
|-- ml_training/
|   |-- generate_training_data.py
|   |-- train_model.py
|   |-- run_full_pipeline.py
|   `-- README.md
|-- android/
|-- ios/
`-- assets/               # includes models/formatting_classifier.tflite
```

## Getting Started

### Prerequisites

- **Flutter SDK**: 3.35.3 or later
- **Dart**: 3.9.2 or later
- **Android Studio** (for Android development)
- **Xcode** (for iOS development - macOS only)

### Installation

1. **Clone the repository**
   ```bash
   cd /path/to/PDM-TI
   cd scan_me_right
   ```

2. **Install dependencies**
   ```bash
   flutter pub get
   ```

3. **Run the app**
   ```bash
   # For Android
   flutter run

   # For specific device
   flutter devices
   flutter run -d <device-id>
   ```

### First Build (Android)

```bash
# Check everything is set up correctly
flutter doctor

# Build APK
flutter build apk --release

# Install on connected device
flutter install
```

## How to Use

1. **Launch the app** - You'll see the home screen with a list of scanned documents
2. **Tap "Scan Document"** - Grant camera permissions when prompted
3. **Capture a photo** - Take a picture of your document or choose from gallery
4. **Review & process** - Optionally add a title, then tap "Process Document"
5. **View results** - See extracted text and formatting
6. **Export** - Share or export as PDF/TXT

## Privacy Features

- Processing is fully offline; no network calls are made
- Documents and metadata remain on-device (SQLite + file storage)
- OCR relies solely on Google ML Kit's on-device APIs
- No analytics, ads, or third-party telemetry SDKs

## Git Workflow

```
main (production releases)
 └── preview (staging/testing)
      ├── feat/camera-capture
      ├── feat/ocr-integration
      ├── feat/ml-formatting
      ├── feat/pdf-export
      └── feat/document-management
```

**Branch Strategy**:
1. Create feature branch from `preview`: `git checkout -b feat/feature-name preview`
2. Develop and commit changes
3. Merge to `preview` for testing
4. Once stable, merge `preview` → `main` for releases

## Testing

```bash
# Run unit tests
flutter test

# Run integration tests
flutter test integration_test/

# Analyze code
flutter analyze
```

## Building for Release

### Android
```bash
# Build APK
flutter build apk --release

# Build App Bundle (for Play Store)
flutter build appbundle --release
```

## Contributing

This is an individual university project for Mobile Devices Programming.

## License

Educational project - developed for PDM (Mobile Devices Programming) course.

## Resources & References

- [Flutter Documentation](https://docs.flutter.dev/)
- [Google ML Kit](https://developers.google.com/ml-kit/vision/text-recognition)
- [TensorFlow Lite](https://www.tensorflow.org/lite)
- [Provider Package](https://pub.dev/packages/provider)
- [SQLite Plugin](https://pub.dev/packages/sqflite)

## Known Issues

- iOS support requires Xcode installation and configuration
- Camera permissions must be granted manually on first use
- ML Kit requires minimum Android SDK 21 (Android 5.0)

## Future Vision

This project aims to create a **commercially viable** offline document scanner that fills a genuine market gap. Unlike existing cloud-based scanners (Microsoft Lens, Adobe Scan, CamScanner), Scan Me Right prioritizes:

1. **Privacy**: No data ever leaves your device
2. **Offline-First**: Works anywhere, no internet needed
3. **Formatting Preservation**: ML-powered style detection
4. **Open Source Potential**: Educational foundation for a real product

---

**Made with ❤️ and Flutter**
