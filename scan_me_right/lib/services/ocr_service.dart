import 'dart:io';
import 'dart:typed_data';
import 'dart:ui' show Rect;

import 'package:google_mlkit_text_recognition/google_mlkit_text_recognition.dart';
import 'package:image/image.dart' as img;

import '../models/document.dart' as model;
import 'formatting_model_service.dart';

/// Service for performing OCR using Google ML Kit
class OCRService {
  OCRService(this._formattingModel);

  final FormattingModelService _formattingModel;
  final TextRecognizer _textRecognizer = TextRecognizer();

  /// Perform OCR on an image file
  Future<String> recognizeText(String imagePath) async {
    final inputImage = InputImage.fromFilePath(imagePath);
    final RecognizedText recognizedText = 
        await _textRecognizer.processImage(inputImage);
    
    return recognizedText.text;
  }

  /// Perform OCR and extract text blocks with positions
  Future<List<model.FormattedTextBlock>> recognizeTextBlocks(String imagePath) async {
    final inputImage = InputImage.fromFilePath(imagePath);
    final RecognizedText recognizedText = 
        await _textRecognizer.processImage(inputImage);

    final originalBytes = await File(imagePath).readAsBytes();
    final originalImage = img.decodeImage(originalBytes);

    
    List<model.FormattedTextBlock> textBlocks = [];
    
    for (TextBlock block in recognizedText.blocks) {
      final formatting = await _predictFormatting(block, originalImage);
      
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

  /// Basic heuristic-based formatting detection
  /// TODO: Replace with ML model for accurate detection
  model.TextFormatting _fallbackFormatting(TextBlock block) {
    final text = block.text;
    final boundingBox = block.boundingBox;

    bool isTitle = false;
    int estimatedFontSize = _estimateFontSize(boundingBox.height);

    if (text == text.toUpperCase() && text.length > 3) {
      isTitle = true;
      estimatedFontSize = (estimatedFontSize + 4).clamp(12, 24);
    }

    return model.TextFormatting(
      isBold: false,
      isItalic: false,
      isUnderlined: false,
      isTitle: isTitle,
      fontSize: estimatedFontSize,
      isBulletList: false,
      isNumberedList: false,
    );
  }

  Future<model.TextFormatting> _predictFormatting(
    TextBlock block,
    img.Image? originalImage,
  ) async {
    if (originalImage == null) {
      return _fallbackFormatting(block);
    }

    final crop = _cropBlock(originalImage, block.boundingBox);
    if (crop == null) {
      return _fallbackFormatting(block);
    }

    try {
      final jpegBytes = Uint8List.fromList(img.encodeJpg(crop, quality: 85));
      final probabilities = await _formattingModel.classifyBlock(jpegBytes);
      return _mapPredictions(probabilities, block.boundingBox);
    } catch (_) {
      return _fallbackFormatting(block);
    }
  }

  img.Image? _cropBlock(img.Image source, Rect rect) {
    final left = rect.left.floor().clamp(0, source.width - 1);
    final top = rect.top.floor().clamp(0, source.height - 1);
    final right = rect.right.ceil().clamp(0, source.width);
    final bottom = rect.bottom.ceil().clamp(0, source.height);

    final width = right - left;
    final height = bottom - top;
    if (width <= 0 || height <= 0) {
      return null;
    }

    return img.copyCrop(
      source,
      x: left,
      y: top,
      width: width,
      height: height,
    );
  }

  model.TextFormatting _mapPredictions(List<double> probs, Rect boundingBox) {
    const threshold = 0.45;

    bool hasBold = probs.length > 2 && probs[2] > threshold;
    bool hasItalic = probs.length > 3 && probs[3] > threshold;
    bool hasUnderline = probs.length > 4 && probs[4] > threshold;
    bool hasTitle = probs.isNotEmpty && probs[0] > threshold;
    bool bullet = probs.length > 5 && probs[5] > threshold;
    bool numbered = probs.length > 6 && probs[6] > threshold;

    final estimatedFontSize = _estimateFontSize(boundingBox.height);

    return model.TextFormatting(
      isBold: hasBold,
      isItalic: hasItalic,
      isUnderlined: hasUnderline,
      isTitle: hasTitle,
      fontSize: estimatedFontSize,
      isBulletList: bullet && !numbered,
      isNumberedList: numbered,
    );
  }

  int _estimateFontSize(double height) {
    if (height >= 140) return 22;
    if (height >= 110) return 20;
    if (height >= 80) return 18;
    if (height >= 60) return 16;
    if (height >= 40) return 14;
    return 12;
  }
 
  /// Dispose the text recognizer
  void dispose() {
    _textRecognizer.close();
  }
}

