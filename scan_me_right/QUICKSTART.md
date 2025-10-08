# Quick Start Guide 🚀

Get your Scan Me Right app running in 5 minutes!

## ✅ Prerequisites (Already Installed)

You have everything you need:
- ✅ Flutter 3.35.3
- ✅ Dart 3.9.2
- ✅ Android SDK
- ✅ Python 3

## 🏃 Run the App

### Step 1: Open Terminal

```bash
cd /Users/tiagomarques/PDM-TI/scan_me_right
```

### Step 2: Run the App

```bash
# Connect your Android device or start an emulator
flutter devices

# Run the app
flutter run
```

That's it! The app should launch on your device.

## 📱 Testing the App

### Test Camera & OCR:
1. Tap the "Scan Document" button
2. Grant camera permission
3. Take a photo of any text (book, printed document, etc.)
4. Tap "Process Document"
5. Wait a few seconds for OCR
6. View the extracted text!

### Test PDF Export:
1. Open a scanned document
2. Tap the menu (⋮) in the top right
3. Select "Export as PDF"
4. Check the notification for file location

### Test Share:
1. Open a document
2. Tap the share icon
3. Share with any app

## 🐛 Common Issues

### "No devices found"
```bash
# Start an Android emulator
flutter emulators
flutter emulators --launch <emulator-name>
```

### "Camera permission denied"
- On your phone: Settings > Apps > Scan Me Right > Permissions > Enable Camera

### "Build failed"
```bash
flutter clean
flutter pub get
flutter run
```

## 🎓 Project Structure

```
lib/
├── main.dart              # App entry point
├── config/                # Configuration & theme
├── models/                # Data models
├── providers/             # State management
├── screens/               # UI screens
│   ├── home_screen.dart
│   ├── camera_screen.dart
│   └── document_detail_screen.dart
├── services/              # Business logic
│   ├── database_service.dart  # SQLite
│   ├── ocr_service.dart      # ML Kit OCR
│   └── pdf_service.dart      # PDF export
└── utils/                 # Helpers
```

## 🔧 Development Commands

```bash
# Hot reload (after changes)
Press 'r' in terminal

# Hot restart
Press 'R' in terminal

# View logs
flutter logs

# Run tests
flutter test

# Check for issues
flutter analyze

# Format code
flutter format lib/
```

## 📊 Next Steps

### For MVP:
- ✅ All core features are implemented
- ✅ OCR works offline
- ✅ PDF/TXT export ready
- ✅ Database storage working

### For Phase 2 (ML Model):
1. Generate training data:
   ```bash
   cd ml_training
   pip install -r requirements.txt
   python generate_training_data.py
   ```
2. Train formatting classifier (see `ml_training/README.md`)
3. Export to TFLite
4. Integrate into app

## 🎯 Key Features Working

- ✅ **Camera Capture** - Take photos or import from gallery
- ✅ **Offline OCR** - Extract text using ML Kit (on-device)
- ✅ **Local Storage** - SQLite database
- ✅ **PDF Export** - Generate styled PDFs
- ✅ **TXT Export** - Plain text export
- ✅ **Document Management** - List, view, delete documents
- ✅ **Share** - Share documents with other apps
- ✅ **Heuristic Formatting** - Basic title/header detection

## 🔒 Privacy Features

- 🔒 100% offline processing
- 🔒 No internet connection needed
- 🔒 No cloud APIs
- 🔒 No data collection
- 🔒 All data stays on device

## 📚 Documentation

- **Full README**: See `README.md`
- **Setup Guide**: See `SETUP_GUIDE.md`
- **ML Training**: See `ml_training/README.md`

## 💡 Tips

1. **Use real documents**: Test with printed text for best OCR results
2. **Good lighting**: OCR works better with clear, well-lit images
3. **Steady hands**: Avoid blurry photos
4. **Flat documents**: Try to capture documents flat, not at angles

## 🆘 Need Help?

Check these resources:
- Project README: `README.md`
- Setup Guide: `SETUP_GUIDE.md`
- Flutter docs: https://docs.flutter.dev
- ML Kit docs: https://developers.google.com/ml-kit

## ✨ You're All Set!

Your app is ready to use. Start scanning documents and enjoy the privacy of offline processing! 🎉

**Happy scanning!** 📸📄

