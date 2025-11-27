import 'package:flutter/foundation.dart';
import 'package:uuid/uuid.dart';
import '../models/document.dart';
import '../services/database_service.dart';
import '../services/ocr_service.dart';
import '../services/pdf_service.dart';
import '../services/formatting_model_service.dart';
import 'dart:io';

/// Provider for managing documents state
class DocumentProvider with ChangeNotifier {
  final DatabaseService _databaseService = DatabaseService.instance;
  FormattingModelService? _formattingModel;
  OCRService? _ocrService;
  final PDFService _pdfService = PDFService();
  
  List<Document> _documents = [];
  bool _isLoading = false;
  String? _error;

  List<Document> get documents => _documents;
  bool get isLoading => _isLoading;
  String? get error => _error;

  /// Load all documents from database
  Future<void> loadDocuments() async {
    _isLoading = true;
    _error = null;
    notifyListeners();

    try {
      _documents = await _databaseService.getAllDocuments();
    } catch (e) {
      _error = 'Failed to load documents: $e';
    } finally {
      _isLoading = false;
      notifyListeners();
    }
  }

  /// Create a new document from an image
  Future<Document?> createDocumentFromImage({
    required String imagePath,
    required String title,
  }) async {
    _isLoading = true;
    _error = null;
    notifyListeners();

    try {
      await init();
      final ocrService = _ocrService;
      if (ocrService == null) {
        throw StateError('DocumentProvider.init() must be called before scanning.');
      }
      // Perform OCR
      final ocrText = await ocrService.recognizeText(imagePath);
      final textBlocks = await ocrService.recognizeTextBlocks(imagePath);

      // Create document
      final document = Document(
        id: const Uuid().v4(),
        title: title,
        imagePath: imagePath,
        ocrText: ocrText,
        createdAt: DateTime.now(),
        updatedAt: DateTime.now(),
        textBlocks: textBlocks,
      );

      // Save to database
      await _databaseService.createDocument(document);
      
      // Add to list
      _documents.insert(0, document);
      
      _isLoading = false;
      notifyListeners();
      
      return document;
    } catch (e) {
      _error = 'Failed to create document: $e';
      _isLoading = false;
      notifyListeners();
      return null;
    }
  }

  /// Update document
  Future<void> updateDocument(Document document) async {
    try {
      await _databaseService.updateDocument(document);
      
      final index = _documents.indexWhere((d) => d.id == document.id);
      if (index != -1) {
        _documents[index] = document;
        notifyListeners();
      }
    } catch (e) {
      _error = 'Failed to update document: $e';
      notifyListeners();
    }
  }

  /// Delete document
  Future<void> deleteDocument(String id) async {
    try {
      await _databaseService.deleteDocument(id);
      _documents.removeWhere((d) => d.id == id);
      notifyListeners();
    } catch (e) {
      _error = 'Failed to delete document: $e';
      notifyListeners();
    }
  }

  /// Export document to PDF
  Future<File?> exportToPDF(Document document) async {
    try {
      return await _pdfService.exportToPDF(document);
    } catch (e) {
      _error = 'Failed to export PDF: $e';
      notifyListeners();
      return null;
    }
  }

  /// Export document to TXT
  Future<File?> exportToTXT(Document document) async {
    try {
      return await _pdfService.exportToTXT(document);
    } catch (e) {
      _error = 'Failed to export TXT: $e';
      notifyListeners();
      return null;
    }
  }

  Future<void> init() async {
    if (_formattingModel != null && _ocrService != null) return;
    _formattingModel = await FormattingModelService.create();
    _ocrService = OCRService(_formattingModel!);
  }

  @override
  void dispose() {
    _ocrService?.dispose();
    _formattingModel?.close();
    super.dispose();
  }
}

