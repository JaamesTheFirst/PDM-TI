# Scan Me Right 📱🔒

**Privacy-first offline document scanner with AI-powered text recognition and formatting preservation.**

An offline-first mobile application that scans documents, extracts text with OCR, preserves formatting (bold, italic, underline, headers), and exports to PDF/TXT — all without sending your sensitive data to the cloud.

## 🎯 Project Goal

Solve the privacy issue: *"When I scan my docs with cloud services, who is actually getting my sensitive information?"*

**Solution**: 100% offline processing. Your documents never leave your device.

## ✨ Features

### MVP Features (Current)
- ✅ **Camera Capture**: Take photos of documents or import from gallery
- ✅ **Offline OCR**: Extract text using Google ML Kit (on-device)
- ✅ **Document Management**: Store and organize scanned documents locally (SQLite)
- ✅ **PDF Export**: Generate styled PDFs offline
- ✅ **TXT Export**: Export plain text
- ✅ **Basic Formatting Detection**: Heuristic-based title/header detection
- ✅ **Share Documents**: Share scanned documents with other apps

### Future Features (Phase 2)
- 🚧 **AI Formatting Recognition**: Custom TensorFlow Lite model for accurate formatting detection
- 🚧 **Bold/Italic/Underline Detection**: ML-powered style classification
- 🚧 **DOCX Export**: Export to Microsoft Word format
- 🚧 **Batch Scanning**: Scan multiple pages into a single document
- 🚧 **Cloud Sync** (Optional): Encrypted backup with NestJS backend

## 🛠 Tech Stack

### Frontend (Mobile App)
- **Framework**: Flutter 3.35+ (Dart 3.9+)
- **State Management**: Provider
- **OCR**: google_mlkit_text_recognition
- **PDF Generation**: pdf + printing packages
- **Local Database**: SQLite (sqflite)
- **Camera**: camera + image_picker
- **Permissions**: permission_handler

### Future: AI Component (Phase 2)
- **Model**: MobileNetV3 (TensorFlow Lite)
- **Training**: Python (TensorFlow, Keras, PIL/OpenCV)
- **Deployment**: TFLite model runs fully offline in Flutter

### Backend (Optional - Phase 3)
- **Framework**: NestJS (TypeScript)
- **Database**: PostgreSQL (Dockerized)
- **Deployment**: Docker containers

## 📁 Project Structure

```
scan_me_right/
├── lib/
│   ├── config/           # App configuration & theme
│   ├── models/           # Data models (Document, TextBlock, etc.)
│   ├── providers/        # State management (DocumentProvider)
│   ├── screens/          # UI screens
│   │   ├── home_screen.dart
│   │   ├── camera_screen.dart
│   │   └── document_detail_screen.dart
│   ├── services/         # Business logic
│   │   ├── database_service.dart
│   │   ├── ocr_service.dart
│   │   └── pdf_service.dart
│   ├── utils/            # Helper utilities
│   ├── widgets/          # Reusable UI components
│   └── main.dart         # App entry point
├── ml_training/          # ML model training scripts
│   ├── generate_training_data.py
│   ├── requirements.txt
│   └── README.md
├── android/              # Android-specific configuration
├── ios/                  # iOS-specific configuration
└── assets/               # Static assets (images, models)
```

## 🚀 Getting Started

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

   # For iOS (macOS only)
   flutter run -d ios

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

## 📱 How to Use

1. **Launch the app** - You'll see the home screen with a list of scanned documents
2. **Tap "Scan Document"** - Grant camera permissions when prompted
3. **Capture a photo** - Take a picture of your document or choose from gallery
4. **Review & process** - Optionally add a title, then tap "Process Document"
5. **View results** - See extracted text and formatting
6. **Export** - Share or export as PDF/TXT

## 🔐 Privacy Features

- ✅ **100% Offline Processing** - No internet connection required
- ✅ **Local Storage Only** - All data stored on your device
- ✅ **No Cloud APIs** - OCR runs entirely on-device with ML Kit
- ✅ **No Analytics** - No tracking or data collection
- ✅ **No Ads** - Clean, privacy-focused experience

## 🌳 Git Workflow

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

## 📊 Development Timeline

### Phase 1: MVP (Weeks 1-4) ✅ COMPLETE
- [x] Flutter project setup
- [x] Camera integration
- [x] ML Kit OCR integration
- [x] Local database (SQLite)
- [x] PDF/TXT export
- [x] Basic UI/UX
- [x] Heuristic formatting detection

### Phase 2: ML Model (Weeks 5-8) 🚧 IN PROGRESS
- [ ] Generate training data
- [ ] Train formatting classifier
- [ ] Export to TensorFlow Lite
- [ ] Integrate TFLite model
- [ ] Replace heuristics with ML predictions

### Phase 3: Polish & Optional Features (Weeks 9-12)
- [ ] Batch scanning
- [ ] DOCX export
- [ ] Cloud sync (optional)
- [ ] App icons & branding
- [ ] App store deployment

## 🧪 Testing

```bash
# Run unit tests
flutter test

# Run integration tests
flutter test integration_test/

# Analyze code
flutter analyze
```

## 📦 Building for Release

### Android
```bash
# Build APK
flutter build apk --release

# Build App Bundle (for Play Store)
flutter build appbundle --release
```

### iOS (macOS only)
```bash
# Build for iOS
flutter build ios --release

# Open in Xcode for signing and deployment
open ios/Runner.xcworkspace
```

## 🤝 Contributing

This is an individual university project for Mobile Devices Programming.

## 📄 License

Educational project - developed for PDM (Mobile Devices Programming) course.

## 👨‍💻 Author

**Tiago Marques**  
Individual Project - Mobile Devices Programming

## 🎓 University Context

- **Course**: Mobile Devices Programming (PDM)
- **Institution**: [Your University]
- **Academic Year**: 2024/2025
- **Project Type**: Individual Assignment

## 📚 Resources & References

- [Flutter Documentation](https://docs.flutter.dev/)
- [Google ML Kit](https://developers.google.com/ml-kit/vision/text-recognition)
- [TensorFlow Lite](https://www.tensorflow.org/lite)
- [Provider Package](https://pub.dev/packages/provider)
- [SQLite Plugin](https://pub.dev/packages/sqflite)

## 🐛 Known Issues

- iOS support requires Xcode installation and configuration
- Camera permissions must be granted manually on first use
- ML Kit requires minimum Android SDK 21 (Android 5.0)

## 🔮 Future Vision

This project aims to create a **commercially viable** offline document scanner that fills a genuine market gap. Unlike existing cloud-based scanners (Microsoft Lens, Adobe Scan, CamScanner), Scan Me Right prioritizes:

1. **Privacy**: No data ever leaves your device
2. **Offline-First**: Works anywhere, no internet needed
3. **Formatting Preservation**: ML-powered style detection
4. **Open Source Potential**: Educational foundation for a real product

---

**Made with ❤️ and Flutter**
