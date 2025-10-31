import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:provider/provider.dart';
import 'package:share_plus/share_plus.dart' show XFile, Share;
import 'dart:io';
import '../models/document.dart';
import '../providers/document_provider.dart';
import 'package:intl/intl.dart';

class DocumentDetailScreen extends StatefulWidget {
  final Document document;

  const DocumentDetailScreen({
    super.key,
    required this.document,
  });

  @override
  State<DocumentDetailScreen> createState() => _DocumentDetailScreenState();
}

class _DocumentDetailScreenState extends State<DocumentDetailScreen> {
  late TextEditingController _textController;
  bool _isEdited = false;

  @override
  void initState() {
    super.initState();
    _textController = TextEditingController(text: widget.document.ocrText ?? '');
    _textController.addListener(() {
      if (_textController.text != widget.document.ocrText) {
        setState(() => _isEdited = true);
      }
    });
  }

  @override
  void dispose() {
    _textController.dispose();
    super.dispose();
  }

  Future<void> _saveChanges() async {
    final updatedDocument = widget.document.copyWith(
      ocrText: _textController.text,
      updatedAt: DateTime.now(),
    );
    
    await context.read<DocumentProvider>().updateDocument(updatedDocument);
    
    if (!mounted) return;
    setState(() => _isEdited = false);
    
    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(
        content: Text('Changes saved!'),
        backgroundColor: Colors.green,
        duration: Duration(seconds: 2),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(widget.document.title),
        actions: [
          IconButton(
            icon: const Icon(Icons.share),
            onPressed: () => _shareDocument(context),
          ),
          PopupMenuButton<String>(
            onSelected: (value) => _handleMenuAction(context, value),
            itemBuilder: (context) => [
              const PopupMenuItem(
                value: 'export_pdf',
                child: Row(
                  children: [
                    Icon(Icons.picture_as_pdf),
                    SizedBox(width: 12),
                    Text('Export as PDF'),
                  ],
                ),
              ),
              const PopupMenuItem(
                value: 'export_txt',
                child: Row(
                  children: [
                    Icon(Icons.text_snippet),
                    SizedBox(width: 12),
                    Text('Export as TXT'),
                  ],
                ),
              ),
              const PopupMenuItem(
                value: 'delete',
                child: Row(
                  children: [
                    Icon(Icons.delete, color: Colors.red),
                    SizedBox(width: 12),
                    Text('Delete', style: TextStyle(color: Colors.red)),
                  ],
                ),
              ),
            ],
          ),
        ],
      ),
      body: SingleChildScrollView(
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // Image Preview
            if (File(widget.document.imagePath).existsSync())
              Container(
                height: 300,
                decoration: BoxDecoration(
                  color: Colors.grey[200],
                ),
                child: Image.file(
                  File(widget.document.imagePath),
                  fit: BoxFit.contain,
                ),
              ),
            
            // Document Info
            Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    'Document Information',
                    style: Theme.of(context).textTheme.titleLarge?.copyWith(
                          fontWeight: FontWeight.bold,
                        ),
                  ),
                  const SizedBox(height: 16),
                  _buildInfoRow(
                    context,
                    'Created',
                    DateFormat('MMMM dd, yyyy • HH:mm').format(widget.document.createdAt),
                  ),
                  const Divider(),
                  
                  // OCR Text
                  const SizedBox(height: 16),
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      Text(
                        'Extracted Text',
                        style: Theme.of(context).textTheme.titleLarge?.copyWith(
                              fontWeight: FontWeight.bold,
                            ),
                      ),
                      Row(
                        children: [
                          if (_isEdited)
                            TextButton.icon(
                              onPressed: _saveChanges,
                              icon: const Icon(Icons.save, size: 20),
                              label: const Text('Save'),
                            ),
                          if (widget.document.ocrText != null)
                            IconButton(
                              icon: const Icon(Icons.copy),
                              tooltip: 'Copy text',
                              onPressed: () => _copyText(context, _textController.text),
                            ),
                        ],
                      ),
                    ],
                  ),
                  const SizedBox(height: 12),
                  
                  if (widget.document.ocrText != null && widget.document.ocrText!.isNotEmpty)
                    Container(
                      padding: const EdgeInsets.all(16),
                      decoration: BoxDecoration(
                        color: Colors.white,
                        borderRadius: BorderRadius.circular(12),
                        border: Border.all(color: Colors.grey[300]!),
                      ),
                      child: TextField(
                        controller: _textController,
                        maxLines: null,
                        style: const TextStyle(
                          fontSize: 16,
                          height: 1.5,
                          color: Colors.black,  // Black text!
                        ),
                        decoration: const InputDecoration(
                          border: InputBorder.none,
                          hintText: 'Edit text...',
                          hintStyle: TextStyle(color: Colors.grey),
                        ),
                      ),
                    )
                  else
                    Container(
                      padding: const EdgeInsets.all(24),
                      decoration: BoxDecoration(
                        color: Colors.grey[100],
                        borderRadius: BorderRadius.circular(12),
                      ),
                      child: Center(
                        child: Text(
                          'No text extracted',
                          style: TextStyle(
                            color: Colors.grey[600],
                            fontStyle: FontStyle.italic,
                          ),
                        ),
                      ),
                    ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildInfoRow(BuildContext context, String label, String value) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 8),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          SizedBox(
            width: 100,
            child: Text(
              label,
              style: TextStyle(
                color: Colors.grey[600],
                fontWeight: FontWeight.w500,
              ),
            ),
          ),
          Expanded(
            child: Text(
              value,
              style: const TextStyle(
                fontWeight: FontWeight.w500,
              ),
            ),
          ),
        ],
      ),
    );
  }

  void _copyText(BuildContext context, String text) {
    Clipboard.setData(ClipboardData(text: text));
    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(
        content: Text('Text copied to clipboard'),
        duration: Duration(seconds: 2),
      ),
    );
  }

  Future<void> _shareDocument(BuildContext context) async {
    try {
      if (File(widget.document.imagePath).existsSync()) {
        await Share.shareXFiles(
          [XFile(widget.document.imagePath)],
          text: widget.document.ocrText ?? widget.document.title,
        );
      }
    } catch (e) {
      if (!context.mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Failed to share: $e')),
      );
    }
  }

  Future<void> _handleMenuAction(BuildContext context, String action) async {
    switch (action) {
      case 'export_pdf':
        await _exportPDF(context);
        break;
      case 'export_txt':
        await _exportTXT(context);
        break;
      case 'delete':
        await _deleteDocument(context);
        break;
    }
  }

  Future<void> _exportPDF(BuildContext context) async {
    try {
      final provider = context.read<DocumentProvider>();
      final file = await provider.exportToPDF(widget.document);
      
      if (!context.mounted) return;
      
      if (file != null) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('PDF saved to ${file.path}'),
            action: SnackBarAction(
              label: 'Share',
              onPressed: () async {
                await Share.shareXFiles([XFile(file.path)]);
              },
            ),
          ),
        );
      }
    } catch (e) {
      if (!context.mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Export failed: $e')),
      );
    }
  }

  Future<void> _exportTXT(BuildContext context) async {
    try {
      final provider = context.read<DocumentProvider>();
      final file = await provider.exportToTXT(widget.document);
      
      if (!context.mounted) return;
      
      if (file != null) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('TXT saved to ${file.path}'),
            action: SnackBarAction(
              label: 'Share',
              onPressed: () async {
                await Share.shareXFiles([XFile(file.path)]);
              },
            ),
          ),
        );
      }
    } catch (e) {
      if (!context.mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Export failed: $e')),
      );
    }
  }

  Future<void> _deleteDocument(BuildContext context) async {
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Delete Document'),
        content: const Text('Are you sure you want to delete this document? This action cannot be undone.'),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context, false),
            child: const Text('Cancel'),
          ),
          TextButton(
            onPressed: () => Navigator.pop(context, true),
            style: TextButton.styleFrom(foregroundColor: Colors.red),
            child: const Text('Delete'),
          ),
        ],
      ),
    );

    if (confirmed == true && context.mounted) {
      await context.read<DocumentProvider>().deleteDocument(widget.document.id);
      
      if (context.mounted) {
        Navigator.pop(context);
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('Document deleted')),
        );
      }
    }
  }
}

