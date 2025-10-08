import 'package:google_mlkit_text_recognition/google_mlkit_text_recognition.dart';
import '../models/document.dart' as model;

/// Service for performing OCR using Google ML Kit
class OCRService {
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
    
    List<model.FormattedTextBlock> textBlocks = [];
    
    for (TextBlock block in recognizedText.blocks) {
      // For now, use basic heuristics for formatting detection
      // Later, this will be replaced with ML model predictions
      final formatting = _detectFormatting(block);
      
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
  model.TextFormatting _detectFormatting(TextBlock block) {
    final text = block.text;
    final boundingBox = block.boundingBox;
    
    // Heuristic: Large text blocks are likely titles
    bool isTitle = false;
    int estimatedFontSize = 12;
    
    final height = boundingBox.height;
    if (height > 50) {
      isTitle = true;
      estimatedFontSize = 18;
    }
    
    // Heuristic: All caps might be a header
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

  /// Dispose the text recognizer
  void dispose() {
    _textRecognizer.close();
  }
}

