# Setup Guide for Mac M3 (Apple Silicon)

Complete setup instructions for developing and running Scan Me Right on Mac Air M3.

## ✅ Prerequisites Check

You already have installed:
- ✅ Flutter 3.35.3
- ✅ Dart 3.9.2
- ✅ Python 3 (for ML training scripts)
- ✅ Android SDK 36.1.0
- ✅ Android Studio 2025.1

## 🔧 Additional Setup Steps

### 1. Flutter Configuration

Your Flutter setup is mostly complete. To fix the Xcode warning (optional, only needed for iOS):

```bash
# Install Xcode from App Store (if you want iOS support)
# Then run:
sudo xcode-select --switch /Applications/Xcode.app/Contents/Developer
sudo xcodebuild -runFirstLaunch
```

**Note**: For MVP, Android development is sufficient. You can skip iOS setup.

### 2. Verify Installation

```bash
cd /Users/tiagomarques/PDM-TI/scan_me_right

# Check Flutter status
flutter doctor -v

# Test dependencies
flutter pub get
```

### 3. Set Up Android Emulator

```bash
# List available emulators
flutter emulators

# Launch an emulator
flutter emulators --launch <emulator-id>

# Or create a new one in Android Studio:
# Tools > Device Manager > Create Device
```

### 4. Connect Physical Device (Recommended)

**For Android**:
1. Enable Developer Options on your phone:
   - Go to Settings > About Phone
   - Tap "Build Number" 7 times
2. Enable USB Debugging:
   - Settings > Developer Options > USB Debugging
3. Connect via USB
4. Accept USB debugging prompt on phone

```bash
# Verify device is connected
flutter devices
```

## 🏃 Running the App

### Option 1: Using Terminal

```bash
cd /Users/tiagomarques/PDM-TI/scan_me_right

# List available devices
flutter devices

# Run on connected device
flutter run

# Run in release mode (faster)
flutter run --release
```

### Option 2: Using VS Code

1. Open project in VS Code
2. Install Flutter extension
3. Open `lib/main.dart`
4. Press F5 or click "Run" in the Debug menu

### Option 3: Using Android Studio

1. Open Android Studio
2. File > Open > Select `scan_me_right` folder
3. Wait for Gradle sync
4. Click the green "Run" button

## 🐛 Troubleshooting

### Issue: "Gradle build failed"

```bash
cd android
./gradlew clean
cd ..
flutter clean
flutter pub get
flutter run
```

### Issue: "Camera permission denied"

The app will prompt for permissions on first use. If denied:
1. Go to phone Settings > Apps > Scan Me Right > Permissions
2. Enable Camera permission

### Issue: "ML Kit not working"

Ensure your Android device/emulator runs Android 5.0+ (API 21+):
```bash
# Check in Android Studio: Tools > Device Manager
# Or check device API level
adb shell getprop ro.build.version.sdk
```

### Issue: "Hot reload not working"

```bash
# Full restart
flutter run --debug
# Press 'R' in terminal to hot restart
```

## 📱 Testing Features

### 1. Test Camera Capture
- Launch app
- Tap "Scan Document"
- Grant camera permission
- Take a photo of printed text
- Verify preview shows

### 2. Test OCR
- After capturing image, tap "Process Document"
- Wait for OCR processing
- Check if extracted text appears

### 3. Test Export
- Open a scanned document
- Tap menu (⋮) > "Export as PDF"
- Check notification for file location
- Verify PDF is created

## 🔍 Checking Logs

```bash
# View real-time logs
flutter logs

# Filter logs
flutter logs | grep "ERROR"

# Android-specific logs
adb logcat | grep flutter
```

## 🧹 Clean Build

If you encounter persistent issues:

```bash
cd /Users/tiagomarques/PDM-TI/scan_me_right

# Deep clean
flutter clean
cd android && ./gradlew clean && cd ..
rm -rf pubspec.lock
rm -rf ios/Pods ios/Podfile.lock

# Reinstall
flutter pub get
flutter pub upgrade
flutter run
```

## 📦 Building Release APK

```bash
cd /Users/tiagomarques/PDM-TI/scan_me_right

# Build release APK
flutter build apk --release

# APK location:
# build/app/outputs/flutter-apk/app-release.apk

# Install on connected device
flutter install
```

## 🐍 Python Setup (For ML Training)

```bash
cd /Users/tiagomarques/PDM-TI/scan_me_right/ml_training

# Create virtual environment (optional but recommended)
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Generate training data
python generate_training_data.py
```

## 📊 Performance Tips

### For Development
```bash
# Run in profile mode for performance profiling
flutter run --profile

# Run with verbose logging
flutter run -v
```

### For Testing
```bash
# Run in release mode for best performance
flutter run --release
```

## 🆘 Getting Help

### Useful Commands
```bash
# Check Flutter version
flutter --version

# Update Flutter
flutter upgrade

# Check for updates
flutter pub outdated

# Analyze code for issues
flutter analyze

# Format code
flutter format lib/
```

### Resources
- Flutter docs: https://docs.flutter.dev
- ML Kit docs: https://developers.google.com/ml-kit
- Stack Overflow: https://stackoverflow.com/questions/tagged/flutter

## ✅ Verification Checklist

Before starting development:
- [ ] `flutter doctor` shows no critical issues
- [ ] `flutter pub get` completes successfully
- [ ] Can launch app on emulator or physical device
- [ ] Camera permission is granted
- [ ] OCR extracts text from test image
- [ ] PDF export works and file is created
- [ ] Python script generates training data

## 🚀 You're Ready!

Your development environment is set up. Next steps:

1. **Run the app**: `flutter run`
2. **Test all features**: Camera, OCR, Export
3. **Make changes**: Edit files in `lib/`
4. **Hot reload**: Press 'r' in terminal or save file
5. **Commit changes**: Use git flow (feat branches → preview → main)

Happy coding! 🎉

