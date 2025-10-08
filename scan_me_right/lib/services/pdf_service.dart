import 'dart:io';
import 'package:pdf/pdf.dart';
import 'package:pdf/widgets.dart' as pw;
import 'package:path_provider/path_provider.dart';
import '../models/document.dart';
import '../config/app_config.dart';

/// Service for exporting documents to PDF
class PDFService {
  /// Export document with OCR text to PDF
  Future<File> exportToPDF(Document document) async {
    final pdf = pw.Document();
    
    // Add page with text content
    pdf.addPage(
      pw.MultiPage(
        pageFormat: PdfPageFormat.a4,
        build: (pw.Context context) {
          return [
            pw.Header(
              level: 0,
              child: pw.Text(
                document.title,
                style: pw.TextStyle(
                  fontSize: 24,
                  fontWeight: pw.FontWeight.bold,
                ),
              ),
            ),
            pw.SizedBox(height: 20),
            _buildTextContent(document),
          ];
        },
      ),
    );
    
    // Save to file
    final directory = await getApplicationDocumentsDirectory();
    final file = File(
      '${directory.path}/${AppConfig.documentsFolder}/${document.title}.pdf',
    );
    
    // Create directory if it doesn't exist
    await file.parent.create(recursive: true);
    
    await file.writeAsBytes(await pdf.save());
    return file;
  }

  /// Build text content with basic formatting
  pw.Widget _buildTextContent(Document document) {
    if (document.textBlocks.isNotEmpty) {
      return pw.Column(
        crossAxisAlignment: pw.CrossAxisAlignment.start,
        children: document.textBlocks.map((block) {
          return pw.Padding(
            padding: const pw.EdgeInsets.only(bottom: 10),
            child: pw.Text(
              block.text,
              style: pw.TextStyle(
                fontSize: block.formatting.isTitle ? 18 : 12,
                fontWeight: block.formatting.isBold 
                    ? pw.FontWeight.bold 
                    : pw.FontWeight.normal,
                fontStyle: block.formatting.isItalic 
                    ? pw.FontStyle.italic 
                    : pw.FontStyle.normal,
              ),
            ),
          );
        }).toList(),
      );
    } else if (document.ocrText != null) {
      return pw.Text(
        document.ocrText!,
        style: const pw.TextStyle(fontSize: 12),
      );
    } else {
      return pw.Text(
        'No text content available',
        style: pw.TextStyle(
          fontSize: 12,
          fontStyle: pw.FontStyle.italic,
          color: PdfColors.grey,
        ),
      );
    }
  }

  /// Export to plain text file
  Future<File> exportToTXT(Document document) async {
    final directory = await getApplicationDocumentsDirectory();
    final file = File(
      '${directory.path}/${AppConfig.documentsFolder}/${document.title}.txt',
    );
    
    // Create directory if it doesn't exist
    await file.parent.create(recursive: true);
    
    String content = '${document.title}\n\n';
    
    if (document.textBlocks.isNotEmpty) {
      for (var block in document.textBlocks) {
        content += '${block.text}\n\n';
      }
    } else if (document.ocrText != null) {
      content += document.ocrText!;
    }
    
    await file.writeAsString(content);
    return file;
  }
}

