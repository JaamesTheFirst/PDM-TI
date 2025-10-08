/// Document model representing a scanned document
class Document {
  final String id;
  final String title;
  final String imagePath;
  final String? ocrText;
  final DateTime createdAt;
  final DateTime updatedAt;
  final List<FormattedTextBlock> textBlocks;

  Document({
    required this.id,
    required this.title,
    required this.imagePath,
    this.ocrText,
    required this.createdAt,
    required this.updatedAt,
    this.textBlocks = const [],
  });

  Map<String, dynamic> toMap() {
    return {
      'id': id,
      'title': title,
      'imagePath': imagePath,
      'ocrText': ocrText,
      'createdAt': createdAt.toIso8601String(),
      'updatedAt': updatedAt.toIso8601String(),
    };
  }

  factory Document.fromMap(Map<String, dynamic> map) {
    return Document(
      id: map['id'],
      title: map['title'],
      imagePath: map['imagePath'],
      ocrText: map['ocrText'],
      createdAt: DateTime.parse(map['createdAt']),
      updatedAt: DateTime.parse(map['updatedAt']),
      textBlocks: [],
    );
  }

  Document copyWith({
    String? title,
    String? imagePath,
    String? ocrText,
    DateTime? updatedAt,
    List<FormattedTextBlock>? textBlocks,
  }) {
    return Document(
      id: id,
      title: title ?? this.title,
      imagePath: imagePath ?? this.imagePath,
      ocrText: ocrText ?? this.ocrText,
      createdAt: createdAt,
      updatedAt: updatedAt ?? this.updatedAt,
      textBlocks: textBlocks ?? this.textBlocks,
    );
  }
}

/// Represents a block of recognized text with potential formatting
class FormattedTextBlock {
  final String text;
  final TextFormatting formatting;
  final BoundingBox boundingBox;

  FormattedTextBlock({
    required this.text,
    required this.formatting,
    required this.boundingBox,
  });
}

/// Text formatting properties (for future ML model)
class TextFormatting {
  final bool isBold;
  final bool isItalic;
  final bool isUnderlined;
  final bool isTitle;
  final int fontSize;

  TextFormatting({
    this.isBold = false,
    this.isItalic = false,
    this.isUnderlined = false,
    this.isTitle = false,
    this.fontSize = 12,
  });
}

/// Bounding box coordinates for text location
class BoundingBox {
  final double x;
  final double y;
  final double width;
  final double height;

  BoundingBox({
    required this.x,
    required this.y,
    required this.width,
    required this.height,
  });
}

