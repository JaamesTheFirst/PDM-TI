#!/bin/bash
# Full ML Training Pipeline
# Runs data generation and model training sequentially

set -e  # Exit on any error

echo "=" | awk '{printf "=%.0s", $(seq 1 60); print ""}'
echo "Document Formatting Recognition - Full Training Pipeline"
echo "=" | awk '{printf "=%.0s", $(seq 1 60); print ""}'
echo ""

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Step 1: Generate Training Data
echo "Step 1/2: Generating Training Data"
echo "-----------------------------------"
echo ""

if [ -d "training_data" ] && [ -f "training_data/metadata.json" ]; then
    echo "⚠️  Training data already exists."
    echo "   To regenerate, delete the training_data/ directory first."
    echo "   Proceeding with existing data..."
    echo ""
else
    echo "Starting data generation..."
    python3 generate_training_data.py
    echo ""
fi

# Step 2: Train Model
echo "Step 2/2: Training Model"
echo "------------------------"
echo ""
echo "Starting model training..."
echo "This will take approximately 2-5 hours on a 16-core CPU."
echo ""

python3 train_model.py

echo ""
echo "=" | awk '{printf "=%.0s", $(seq 1 60); print ""}'
echo "✅ Pipeline Complete!"
echo "=" | awk '{printf "=%.0s", $(seq 1 60); print ""}'
echo ""
echo "Results:"
echo "  - Best model: models/best_model.keras"
echo "  - TFLite model: models/formatting_classifier.tflite"
echo "  - Training plots: models/training_history.png"
echo ""

