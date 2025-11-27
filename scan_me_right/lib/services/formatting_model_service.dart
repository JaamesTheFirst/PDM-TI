import 'dart:typed_data';

import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';

class FormattingModelService {
  FormattingModelService._(this._interpreter);

  static const _modelPath = 'assets/models/formatting_classifier.tflite';
  static const _inputSize = 224;
  static const _numClasses = 7;

  static Future<FormattingModelService> create() async {
    final interpreter = await Interpreter.fromAsset(_modelPath);
    return FormattingModelService._(interpreter);
  }

  final Interpreter _interpreter;

  Future<List<double>> classifyBlock(Uint8List jpegBytes) async {
    final img.Image? original = img.decodeImage(jpegBytes);
    if (original == null) {
      throw Exception('Failed to decode block image');
    }

    final img.Image resized =
        img.copyResize(original, width: _inputSize, height: _inputSize);

    final input = List.generate(
      _inputSize,
      (_) => List.generate(
        _inputSize,
        (_) => List.filled(3, 0.0),
      ),
    );

    for (var y = 0; y < _inputSize; y++) {
      for (var x = 0; x < _inputSize; x++) {
        final pixel = resized.getPixel(x, y);
        input[y][x][0] = pixel.r / 255.0;
        input[y][x][1] = pixel.g / 255.0;
        input[y][x][2] = pixel.b / 255.0;
      }
    }

    final output = List.filled(_numClasses, 0.0).reshape([1, _numClasses]);
    _interpreter.run([input], output);

    final raw = output[0].cast<double>();
    return raw.map((value) => value.clamp(0.0, 1.0)).toList();
  }

  void close() => _interpreter.close();
}
