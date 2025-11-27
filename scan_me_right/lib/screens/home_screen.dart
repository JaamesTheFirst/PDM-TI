import 'dart:async';

import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../providers/document_provider.dart';
import '../config/app_config.dart';
import 'camera_screen.dart';
import 'document_detail_screen.dart';
import 'package:intl/intl.dart';
import '../models/document.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  String? _recentDocumentId;
  bool _highlightPulseOn = false;
  Timer? _highlightTimer;

  @override
  void initState() {
    super.initState();
    // Load documents when screen initializes
    WidgetsBinding.instance.addPostFrameCallback((_) {
      final provider = context.read<DocumentProvider>();
      provider.init().then((_) => provider.loadDocuments());
    });
  }

  @override
  void dispose() {
    _highlightTimer?.cancel();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text(AppConfig.appName),
        actions: [
          IconButton(
            icon: const Icon(Icons.info_outline),
            onPressed: () => _showAboutDialog(context),
          ),
        ],
      ),
      body: Consumer<DocumentProvider>(
        builder: (context, provider, child) {
          if (provider.isLoading && provider.documents.isEmpty) {
            return const Center(child: CircularProgressIndicator());
          }

          if (provider.documents.isEmpty) {
            return _buildEmptyState(context);
          }

          return RefreshIndicator(
            onRefresh: () => provider.loadDocuments(),
            child: ListView.builder(
              padding: const EdgeInsets.all(16),
              itemCount: provider.documents.length,
              itemBuilder: (context, index) {
                final document = provider.documents[index];
                return _buildDocumentCard(context, document);
              },
            ),
          );
        },
      ),
      floatingActionButton: FloatingActionButton.extended(
        onPressed: () => _navigateToCamera(context),
        icon: const Icon(Icons.camera_alt),
        label: const Text('Scan Document'),
      ),
    );
  }

  Widget _buildEmptyState(BuildContext context) {
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Icon(
            Icons.document_scanner_outlined,
            size: 120,
            color: Colors.grey[300],
          ),
          const SizedBox(height: 24),
          Text(
            'No Documents Yet',
            style: Theme.of(context).textTheme.headlineSmall?.copyWith(
                  color: Colors.grey[600],
                  fontWeight: FontWeight.bold,
                ),
          ),
          const SizedBox(height: 12),
          Text(
            'Tap the button below to scan your first document',
            style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                  color: Colors.grey[500],
                ),
            textAlign: TextAlign.center,
          ),
          const SizedBox(height: 32),
          ElevatedButton.icon(
            onPressed: () => _navigateToCamera(context),
            icon: const Icon(Icons.camera_alt),
            label: const Text('Scan Document'),
            style: ElevatedButton.styleFrom(
              padding: const EdgeInsets.symmetric(horizontal: 32, vertical: 16),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildDocumentCard(BuildContext context, Document document) {
    final isHighlighted =
        _recentDocumentId == document.id && _highlightPulseOn;
    final baseColor = Theme.of(context).cardColor;
    final highlightColor = AppConfig.primaryColor.withOpacity(0.15);

    return AnimatedContainer(
      duration: const Duration(milliseconds: 220),
      curve: Curves.easeInOut,
      margin: const EdgeInsets.only(bottom: 12),
      decoration: BoxDecoration(
        color: isHighlighted ? highlightColor : baseColor,
        borderRadius: BorderRadius.circular(12),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(
              isHighlighted ? 0.18 : 0.06,
            ),
            blurRadius: isHighlighted ? 18 : 10,
            offset: const Offset(0, 6),
          ),
        ],
      ),
      child: Material(
        color: Colors.transparent,
        child: ListTile(
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(12),
          ),
          contentPadding: const EdgeInsets.all(16),
          leading: CircleAvatar(
            radius: 30,
            backgroundColor: AppConfig.primaryColor.withOpacity(0.1),
            child: Icon(
              Icons.description,
              color: AppConfig.primaryColor,
              size: 30,
            ),
          ),
          title: Text(
            document.title,
            style: const TextStyle(
              fontWeight: FontWeight.bold,
              fontSize: 16,
            ),
          ),
          subtitle: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const SizedBox(height: 4),
              Text(
                DateFormat('MMM dd, yyyy • HH:mm').format(document.createdAt),
                style: TextStyle(color: Colors.grey[600]),
              ),
              if (document.ocrText != null && document.ocrText!.isNotEmpty) ...[
                const SizedBox(height: 4),
                Text(
                  document.ocrText!.length > 50
                      ? '${document.ocrText!.substring(0, 50)}...'
                      : document.ocrText!,
                  maxLines: 2,
                  overflow: TextOverflow.ellipsis,
                  style: const TextStyle(
                    color: Colors.black87,
                    fontSize: 12,
                  ),
                ),
              ],
            ],
          ),
          trailing: const Icon(Icons.chevron_right),
          onTap: () => _navigateToDetail(context, document),
          onLongPress: () => _confirmDeleteDocument(context, document),
        ),
      ),
    );
  }

  Future<void> _navigateToCamera(BuildContext context) async {
    final newDocumentId = await Navigator.push<String?>(
      context,
      MaterialPageRoute(builder: (context) => const CameraScreen()),
    );

    if (!mounted) return;
    if (newDocumentId != null) {
      _triggerHighlight(newDocumentId);
    }
  }

  void _navigateToDetail(BuildContext context, document) {
    Navigator.push(
      context,
      MaterialPageRoute(
        builder: (context) => DocumentDetailScreen(document: document),
      ),
    );
  }

  void _showAboutDialog(BuildContext context) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text(AppConfig.appName),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('Version ${AppConfig.appVersion}'),
            const SizedBox(height: 12),
            const Text(AppConfig.appDescription),
            const SizedBox(height: 12),
            const Text(
              '🔒 100% Offline - Your data never leaves your device',
              style: TextStyle(fontWeight: FontWeight.bold),
            ),
          ],
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Close'),
          ),
        ],
      ),
    );
  }

  void _triggerHighlight(String documentId) {
    _highlightTimer?.cancel();
    setState(() {
      _recentDocumentId = documentId;
      _highlightPulseOn = true;
    });

    int toggles = 0;
    _highlightTimer = Timer.periodic(
      const Duration(milliseconds: 320),
      (timer) {
        if (!mounted) {
          timer.cancel();
          return;
        }
        toggles++;
        setState(() {
          _highlightPulseOn = !_highlightPulseOn;
        });
        if (toggles >= 4) {
          timer.cancel();
          setState(() {
            _highlightPulseOn = false;
            _recentDocumentId = null;
          });
        }
      },
    );
  }

  Future<void> _confirmDeleteDocument(
    BuildContext context,
    Document document,
  ) async {
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Delete Document'),
        content: const Text(
          'This document will be removed permanently. Do you want to continue?',
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context, false),
            child: const Text('Cancel'),
          ),
          TextButton(
            onPressed: () => Navigator.pop(context, true),
            style: TextButton.styleFrom(
              foregroundColor: Colors.red,
            ),
            child: const Text('Delete'),
          ),
        ],
      ),
    );

    if (confirmed == true) {
      await context.read<DocumentProvider>().deleteDocument(document.id);
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('Document deleted'),
          duration: Duration(seconds: 2),
        ),
      );
    }
  }
}

