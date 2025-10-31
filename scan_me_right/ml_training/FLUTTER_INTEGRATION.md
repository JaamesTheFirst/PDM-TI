# 🔌 Flutter Integration Guide

After training your model, follow these steps to integrate it into your app.

---

## Step 1: Copy Model to Assets

```bash
# From ml_training directory
cp formatting_classifier.tflite ../assets/models/
```

---

## Step 2: Update `pubspec.yaml`

Edit `scan_me_right/pubspec.yaml`:

### Uncomment TFLite dependency:
```yaml
dependencies:
  # Uncomment this line:
  tflite_flutter: ^0.10.4
```

### Uncomment asset path:
```yaml
flutter:
  assets:
    # Uncomment this line:
    - assets/models/formatting_classifier.tflite
```

### Install dependency:
```bash
cd /Users/tiagomarques/PDM-TI/scan_me_right
flutter pub get
```

---

## Step 3: Create TFLite Service

Create new file: `lib/services/tflite_service.dart`

```dart
import 'dart:typed_data';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;
import 'dart:io';

class TFLiteService {
  Interpreter? _interpreter;
  
  final List<String> _labels = [
    'normal',
    'bold', 
    'italic',
    'underline',
    'title'
  ];

  Future<void> loadModel() async {
    try {
      _interpreter = await Interpreter.fromAsset(
        'assets/models/formatting_classifier.tflite'
      );
      print('✅ TFLite model loaded successfully');
    } catch (e) {
      print('❌ Error loading model: $e');
    }
  }

  Future<Map<String, dynamic>> classifyTextRegion(
    String imagePath,
    int x,
    int y,
    int width,
    int height,
  ) async {
    if (_interpreter == null) {
      await loadModel();
    }

    try {
      // Load and crop image to text region
      final imageBytes = await File(imagePath).readAsBytes();
      final image = img.decodeImage(imageBytes);
      
      if (image == null) {
        throw Exception('Failed to decode image');
      }

      // Crop to text region
      final cropped = img.copyCrop(
        image,
        x: x,
        y: y,
        width: width,
        height: height,
      );

      // Resize to model input size (224x224)
      final resized = img.copyResize(
        cropped,
        width: 224,
        height: 224,
      );

      // Convert to float array and normalize
      final input = _imageToByteList(resized);

      // Run inference
      var output = List.filled(5, 0.0).reshape([1, 5]);
      _interpreter!.run(input, output);

      // Get predictions
      final probabilities = output[0] as List<double>;
      final maxIndex = probabilities.indexOf(
        probabilities.reduce((a, b) => a > b ? a : b)
      );
      final confidence = probabilities[maxIndex];

      return {
        'class': _labels[maxIndex],
        'confidence': confidence,
        'is_bold': _labels[maxIndex] == 'bold',
        'is_italic': _labels[maxIndex] == 'italic',
        'is_underlined': _labels[maxIndex] == 'underline',
        'is_title': _labels[maxIndex] == 'title',
      };
    } catch (e) {
      print('Error classifying: $e');
      return {
        'class': 'normal',
        'confidence': 0.0,
        'is_bold': false,
        'is_italic': false,
        'is_underlined': false,
        'is_title': false,
      };
    }
  }

  Float32List _imageToByteList(img.Image image) {
    var convertedBytes = Float32List(1 * 224 * 224 * 3);
    var buffer = Float32List.view(convertedBytes.buffer);
    int pixelIndex = 0;

    for (int i = 0; i < 224; i++) {
      for (int j = 0; j < 224; j++) {
        var pixel = image.getPixel(j, i);
        buffer[pixelIndex++] = pixel.r / 255.0;
        buffer[pixelIndex++] = pixel.g / 255.0;
        buffer[pixelIndex++] = pixel.b / 255.0;
      }
    }

    return convertedBytes.reshape([1, 224, 224, 3]);
  }

  void dispose() {
    _interpreter?.close();
  }
}
```

---

## Step 4: Update `ocr_service.dart`

Replace the heuristic detection with ML model:

```dart
import 'package:google_mlkit_text_recognition/google_mlkit_text_recognition.dart';
import '../models/document.dart' as model;
import 'tflite_service.dart';  // Add this import

class OCRService {
  final TextRecognizer _textRecognizer = TextRecognizer();
  final TFLiteService _tfliteService = TFLiteService();  // Add this
  bool _modelLoaded = false;

  // Add this method
  Future<void> initialize() async {
    if (!_modelLoaded) {
      await _tfliteService.loadModel();
      _modelLoaded = true;
    }
  }

  Future<List<model.FormattedTextBlock>> recognizeTextBlocks(String imagePath) async {
    // Initialize model if not loaded
    await initialize();
    
    final inputImage = InputImage.fromFilePath(imagePath);
    final RecognizedText recognizedText = 
        await _textRecognizer.processImage(inputImage);
    
    List<model.FormattedTextBlock> textBlocks = [];
    
    for (TextBlock block in recognizedText.blocks) {
      // Use ML model for formatting detection!
      final formatting = await _detectFormattingML(imagePath, block);
      
      final boundingBox = model.BoundingBox(
        x: block.boundingBox.left.toDouble(),
        y: block.boundingBox.top.toDouble(),
        width: block.boundingBox.width.toDouble(),
        height: block.boundingBox.height.toDouble(),
      );
      
      textBlocks.add(
        model.FormattedTextBlock(
          text: block.text,
          formatting: formatting,
          boundingBox: boundingBox,
        ),
      );
    }
    
    return textBlocks;
  }

  // NEW: ML-based formatting detection
  Future<model.TextFormatting> _detectFormattingML(
    String imagePath,
    TextBlock block,
  ) async {
    try {
      final bbox = block.boundingBox;
      
      // Use TFLite model to classify
      final result = await _tfliteService.classifyTextRegion(
        imagePath,
        bbox.left,
        bbox.top,
        bbox.width,
        bbox.height,
      );
      
      // Determine font size based on bounding box
      int fontSize = 12;
      if (result['is_title']) {
        fontSize = 18;
      } else if (result['is_bold']) {
        fontSize = 14;
      }
      
      return model.TextFormatting(
        isBold: result['is_bold'] || result['is_title'],
        isItalic: result['is_italic'],
        isUnderlined: result['is_underlined'],
        isTitle: result['is_title'],
        fontSize: fontSize,
      );
    } catch (e) {
      print('ML classification failed, using fallback: $e');
      // Fallback to heuristics if ML fails
      return _detectFormattingHeuristic(block);
    }
  }

  // Keep old heuristic method as fallback
  model.TextFormatting _detectFormattingHeuristic(TextBlock block) {
    // Your existing heuristic code...
    final text = block.text;
    final boundingBox = block.boundingBox;
    
    bool isTitle = false;
    int estimatedFontSize = 12;
    
    final height = boundingBox.height;
    if (height > 50) {
      isTitle = true;
      estimatedFontSize = 18;
    }
    
    if (text == text.toUpperCase() && text.length > 3) {
      isTitle = true;
    }
    
    return model.TextFormatting(
      isBold: false,
      isItalic: false,
      isUnderlined: false,
      isTitle: isTitle,
      fontSize: estimatedFontSize,
    );
  }

  void dispose() {
    _textRecognizer.close();
    _tfliteService.dispose();  // Add this
  }
}
```

---

## Step 6: Test the Integration

### Run the app:
```bash
cd /Users/tiagomarques/PDM-TI/scan_me_right
flutter run
```

### Test formatting detection:
1. Scan a document with **bold titles**
2. Check if OCR detects formatting
3. Export to PDF - formatting should appear!

---

## 📊 Performance Expectations

### Model Performance:
- **Inference time**: 50-150ms per text block
- **Model size**: ~4-8 MB
- **Accuracy**: 85-95% (depends on training data quality)
- **Memory**: Minimal (optimized for mobile)

### What Works Well:
- ✅ Title vs Normal text (easiest)
- ✅ Bold vs Normal (good)
- ⚠️ Italic detection (moderate - synthetic data limitation)
- ⚠️ Underline (moderate - synthetic data limitation)

### Improvement Ideas:
1. Add more training samples
2. Use real document images (download datasets)
3. Add font variations in data generation
4. Train longer (more epochs)

---

## 🎯 Success Criteria

Your model is ready when:
- [x] Test accuracy > 85%
- [x] Model size < 10 MB
- [x] Inference runs without errors
- [x] PDF export shows correct formatting
- [x] App doesn't crash or slow down

---

## 🔄 Iteration Cycle

```
Generate Data → Train → Test → Integrate → Deploy
                  ↑                            ↓
                  └────────── Improve ─────────┘
```

If accuracy is low:
1. Generate more diverse data
2. Add data augmentation
3. Try different model architectures
4. Collect real document images

---

**Ready to train? Run the commands in order!** 🚀

