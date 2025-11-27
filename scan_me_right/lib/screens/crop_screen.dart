import 'dart:typed_data';

import 'package:crop_your_image/crop_your_image.dart';
import 'package:flutter/material.dart';

class CropImageScreen extends StatefulWidget {
  const CropImageScreen({
    super.key,
    required this.imageBytes,
  });

  final Uint8List imageBytes;

  @override
  State<CropImageScreen> createState() => _CropImageScreenState();
}

class _CropImageScreenState extends State<CropImageScreen> {
  final CropController _controller = CropController();
  bool _isCropping = false;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Crop Image'),
        actions: [
          TextButton(
            onPressed: _isCropping ? null : _onCropPressed,
            child: const Text('Done'),
          ),
        ],
      ),
      body: Stack(
        children: [
          Positioned.fill(
            child: Crop(
              controller: _controller,
              image: widget.imageBytes,
              withCircleUi: false,
              baseColor: Colors.black,
              maskColor: Colors.black.withOpacity(0.4),
              interactive: true,
              onCropped: (cropResult) {
                if (!mounted) return;
                setState(() {
                  _isCropping = false;
                });
                if (cropResult is CropSuccess) {
                  Navigator.pop(context, cropResult.croppedImage);
                } else if (cropResult is CropFailure) {
                  ScaffoldMessenger.of(context).showSnackBar(
                    SnackBar(
                      content: Text('Failed to crop image: ${cropResult.cause}'),
                    ),
                  );
                }
              },
            ),
          ),
          if (_isCropping)
            const Positioned.fill(
              child: ColoredBox(
                color: Color.fromARGB(120, 0, 0, 0),
                child: Center(
                  child: CircularProgressIndicator(),
                ),
              ),
            ),
        ],
      ),
      bottomNavigationBar: SafeArea(
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 16),
          child: Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              OutlinedButton(
                onPressed: _isCropping ? null : () => Navigator.pop(context),
                style: OutlinedButton.styleFrom(
                  padding: const EdgeInsets.symmetric(
                    horizontal: 20,
                    vertical: 12,
                  ),
                  textStyle: Theme.of(context)
                      .textTheme
                      .labelLarge
                      ?.copyWith(fontSize: 14),
                ),
                child: const Text('Cancel'),
              ),
              FilledButton(
                onPressed: _isCropping ? null : _onCropPressed,
                style: FilledButton.styleFrom(
                  padding: const EdgeInsets.symmetric(
                    horizontal: 20,
                    vertical: 12,
                  ),
                  textStyle: Theme.of(context)
                      .textTheme
                      .labelLarge
                      ?.copyWith(fontSize: 14),
                ),
                child: const Text('Apply Crop'),
              ),
            ],
          ),
        ),
      ),
    );
  }

  void _onCropPressed() {
    setState(() {
      _isCropping = true;
    });
    _controller.crop();
  }
}

