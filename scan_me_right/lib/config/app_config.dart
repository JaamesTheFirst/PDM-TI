import 'package:flutter/material.dart';

/// Application-wide configuration
class AppConfig {
  // App Info
  static const String appName = 'Scan Me Right';
  static const String appVersion = '1.0.0';
  static const String appDescription = 
      'Privacy-first offline document scanner';

  // Colors
  static const Color primaryColor = Color(0xFF2196F3);
  static const Color accentColor = Color(0xFF03A9F4);
  static const Color errorColor = Color(0xFFE57373);
  static const Color successColor = Color(0xFF81C784);
  
  // Database
  static const String databaseName = 'scan_me_right.db';
  static const int databaseVersion = 1;
  
  // Storage
  static const String documentsFolder = 'scanned_documents';
  static const String imagesFolder = 'document_images';
  
  // OCR Settings
  static const double minTextConfidence = 0.5;
  
  // Export Settings
  static const String defaultPdfName = 'Scanned_Document';
  static const String defaultTxtName = 'Extracted_Text';
}

/// Theme configuration
class AppTheme {
  static ThemeData lightTheme = ThemeData(
    useMaterial3: true,
    colorScheme: ColorScheme.fromSeed(
      seedColor: AppConfig.primaryColor,
      brightness: Brightness.light,
    ),
    appBarTheme: const AppBarTheme(
      centerTitle: true,
      elevation: 0,
    ),
    floatingActionButtonTheme: FloatingActionButtonThemeData(
      backgroundColor: AppConfig.primaryColor,
      foregroundColor: Colors.white,
    ),
    cardTheme: CardThemeData(
      elevation: 2,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(12),
      ),
    ),
  );
  
  static ThemeData darkTheme = ThemeData(
    useMaterial3: true,
    colorScheme: ColorScheme.fromSeed(
      seedColor: AppConfig.primaryColor,
      brightness: Brightness.dark,
    ),
    appBarTheme: const AppBarTheme(
      centerTitle: true,
      elevation: 0,
    ),
    floatingActionButtonTheme: FloatingActionButtonThemeData(
      backgroundColor: AppConfig.primaryColor,
      foregroundColor: Colors.white,
    ),
    cardTheme: CardThemeData(
      elevation: 2,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(12),
      ),
    ),
  );
}

