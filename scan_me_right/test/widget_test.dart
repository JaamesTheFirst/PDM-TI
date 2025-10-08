// Basic widget test for Scan Me Right app

import 'package:flutter_test/flutter_test.dart';

import 'package:scan_me_right/main.dart';

void main() {
  testWidgets('App launches successfully', (WidgetTester tester) async {
    // Build our app and trigger a frame.
    await tester.pumpWidget(const ScanMeRightApp());

    // Verify that the app title is displayed
    expect(find.text('Scan Me Right'), findsOneWidget);
    
    // Verify that the scan button is present
    expect(find.text('Scan Document'), findsWidgets);
  });
}
