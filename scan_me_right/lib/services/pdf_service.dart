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
      final children = <pw.Widget>[];
      int orderedIndex = 1;

      for (final block in document.textBlocks) {
        final formatting = block.formatting;
        final textStyle = pw.TextStyle(
          fontSize: formatting.isTitle ? 20 : formatting.fontSize.toDouble(),
          fontWeight: formatting.isBold || formatting.isTitle
              ? pw.FontWeight.bold
              : pw.FontWeight.normal,
          fontStyle:
              formatting.isItalic ? pw.FontStyle.italic : pw.FontStyle.normal,
          decoration: formatting.isUnderlined
              ? pw.TextDecoration.underline
              : pw.TextDecoration.none,
        );

        pw.Widget line;
        if (formatting.isBulletList) {
          orderedIndex = 1;
          line = pw.Row(
            crossAxisAlignment: pw.CrossAxisAlignment.start,
            children: [
              pw.Padding(
                padding: const pw.EdgeInsets.only(right: 8, top: 2),
                child: pw.Text('•', style: pw.TextStyle(fontSize: textStyle.fontSize)),
              ),
              pw.Expanded(
                child: pw.Text(block.text, style: textStyle),
              ),
            ],
          );
        } else if (formatting.isNumberedList) {
          line = pw.Row(
            crossAxisAlignment: pw.CrossAxisAlignment.start,
            children: [
              pw.Padding(
                padding: const pw.EdgeInsets.only(right: 8, top: 2),
                child: pw.Text('${orderedIndex++}.',
                    style: pw.TextStyle(
                      fontSize: textStyle.fontSize,
                      fontWeight: pw.FontWeight.bold,
                    )),
              ),
              pw.Expanded(
                child: pw.Text(block.text, style: textStyle),
              ),
            ],
          );
        } else {
          orderedIndex = 1;
          line = pw.Text(block.text, style: textStyle);
        }

        children.add(
          pw.Padding(
            padding: const pw.EdgeInsets.only(bottom: 10),
            child: line,
          ),
        );
      }

      return pw.Column(
        crossAxisAlignment: pw.CrossAxisAlignment.start,
        children: children,
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
      int orderedIndex = 1;
      for (final block in document.textBlocks) {
        final formatting = block.formatting;
        if (formatting.isBulletList) {
          orderedIndex = 1;
          content += '- ${block.text}\n\n';
        } else if (formatting.isNumberedList) {
          content += '${orderedIndex++}. ${block.text}\n\n';
        } else {
          orderedIndex = 1;
          content += '${block.text}\n\n';
        }
      }
    } else if (document.ocrText != null) {
      content += document.ocrText!;
    }
    
    await file.writeAsString(content);
    return file;
  }
}

