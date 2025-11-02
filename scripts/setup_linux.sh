#!/usr/bin/env bash
set -euo pipefail

# Minimal Linux setup for Scan Me Right (Flutter + Python ML deps)

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
APT_REQS_FILE="$PROJECT_ROOT/scripts/requirements.apt"
FLUTTER_VERSION="3.35.3"
FLUTTER_TARBALL="flutter_linux_${FLUTTER_VERSION}-stable.tar.xz"
FLUTTER_URL="https://storage.googleapis.com/flutter_infra_release/releases/stable/linux/${FLUTTER_TARBALL}"
FLUTTER_HOME="$HOME/flutter"

echo "[1/6] Installing system packages via apt..."
if command -v apt >/dev/null 2>&1; then
    sudo apt update
    xargs -a "$APT_REQS_FILE" sudo apt install -y --no-install-recommends
else
    echo "This script currently supports apt-based systems only. Please install listed packages manually." >&2
    exit 1
fi

echo "[2/6] Installing Flutter SDK ${FLUTTER_VERSION}..."
if [ ! -d "$FLUTTER_HOME" ]; then
    mkdir -p "$HOME"
    cd "$HOME"
    curl -LO "$FLUTTER_URL"
    tar xf "$FLUTTER_TARBALL"
    rm -f "$FLUTTER_TARBALL"
else
    echo "Flutter appears to be installed at $FLUTTER_HOME; skipping download."
fi

if ! command -v flutter >/dev/null 2>&1; then
    echo "[3/6] Adding Flutter to PATH in ~/.bashrc..."
    if ! grep -q "export FLUTTER_HOME=\$HOME/flutter" "$HOME/.bashrc" 2>/dev/null; then
        {
            echo "export FLUTTER_HOME=\$HOME/flutter"
            echo "export PATH=\$FLUTTER_HOME/bin:\$PATH"
        } >> "$HOME/.bashrc"
    fi
    export FLUTTER_HOME="$FLUTTER_HOME"
    export PATH="$FLUTTER_HOME/bin:$PATH"
fi

echo "[4/6] Running flutter doctor and fetching Dart packages..."
flutter --version || true
flutter doctor -v || true
cd "$PROJECT_ROOT/scan_me_right"
flutter pub get

echo "[5/6] (Optional) Set up Python venv and ML requirements..."
cd "$PROJECT_ROOT/scan_me_right/ml_training"
python3 -m venv venv || true
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
deactivate

echo "[6/6] Done. Next steps:"
cat <<'NEXT'
- Open a new shell or run: source ~/.bashrc
- Connect an Android device or start an emulator
- From scan_me_right/: flutter run

If Android SDK/Studio is missing, install Android Studio and re-run "flutter doctor".
NEXT

echo "Setup completed successfully."


