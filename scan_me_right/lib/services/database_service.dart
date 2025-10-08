import 'package:sqflite/sqflite.dart';
import 'package:path/path.dart';
import '../models/document.dart';
import '../config/app_config.dart';

/// Service for managing local SQLite database
class DatabaseService {
  static final DatabaseService instance = DatabaseService._init();
  static Database? _database;

  DatabaseService._init();

  Future<Database> get database async {
    if (_database != null) return _database!;
    _database = await _initDB();
    return _database!;
  }

  Future<Database> _initDB() async {
    final dbPath = await getDatabasesPath();
    final path = join(dbPath, AppConfig.databaseName);

    return await openDatabase(
      path,
      version: AppConfig.databaseVersion,
      onCreate: _createDB,
    );
  }

  Future<void> _createDB(Database db, int version) async {
    await db.execute('''
      CREATE TABLE documents (
        id TEXT PRIMARY KEY,
        title TEXT NOT NULL,
        imagePath TEXT NOT NULL,
        ocrText TEXT,
        createdAt TEXT NOT NULL,
        updatedAt TEXT NOT NULL
      )
    ''');
  }

  // Create
  Future<Document> createDocument(Document document) async {
    final db = await database;
    await db.insert('documents', document.toMap());
    return document;
  }

  // Read all
  Future<List<Document>> getAllDocuments() async {
    final db = await database;
    final result = await db.query(
      'documents',
      orderBy: 'createdAt DESC',
    );
    return result.map((map) => Document.fromMap(map)).toList();
  }

  // Read by ID
  Future<Document?> getDocumentById(String id) async {
    final db = await database;
    final result = await db.query(
      'documents',
      where: 'id = ?',
      whereArgs: [id],
    );
    
    if (result.isNotEmpty) {
      return Document.fromMap(result.first);
    }
    return null;
  }

  // Update
  Future<int> updateDocument(Document document) async {
    final db = await database;
    return db.update(
      'documents',
      document.toMap(),
      where: 'id = ?',
      whereArgs: [document.id],
    );
  }

  // Delete
  Future<int> deleteDocument(String id) async {
    final db = await database;
    return db.delete(
      'documents',
      where: 'id = ?',
      whereArgs: [id],
    );
  }

  // Close database
  Future<void> close() async {
    final db = await database;
    db.close();
  }
}

