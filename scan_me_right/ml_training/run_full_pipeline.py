#!/usr/bin/env python3
"""
Full ML Training Pipeline
Runs data generation and model training sequentially
"""

import os
import sys
import subprocess

def run_command(cmd, description):
    """Run a command and handle errors"""
    print("=" * 60)
    print(description)
    print("=" * 60)
    print()
    
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            check=True,
            capture_output=False,
            text=True
        )
        print()
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error: {description} failed!")
        print(f"Exit code: {e.returncode}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
        sys.exit(130)

def main():
    """Main pipeline execution"""
    print("=" * 60)
    print("Document Formatting Recognition - Full Training Pipeline")
    print("=" * 60)
    print()
    
    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Step 1: Generate Training Data
    training_data_dir = "training_data"
    metadata_file = os.path.join(training_data_dir, "metadata.json")
    
    if os.path.exists(metadata_file):
        print("⚠️  Training data already exists.")
        print("   To regenerate, delete the training_data/ directory first.")
        print("   Proceeding with existing data...")
        print()
    else:
        run_command(
            "python3 generate_training_data.py",
            "Step 1/2: Generating Training Data"
        )
    
    # Step 2: Train Model
    print()
    print("⚠️  NOTE: Training will take approximately 2-5 hours on 16-core CPU")
    print("   You can safely leave this running overnight.")
    print()
    
    run_command(
        "python3 train_model.py",
        "Step 2/2: Training Model"
    )
    
    # Summary
    print()
    print("=" * 60)
    print("✅ Pipeline Complete!")
    print("=" * 60)
    print()
    print("Results:")
    print("  - Best model: models/best_model.keras")
    print("  - TFLite model: models/formatting_classifier.tflite")
    print("  - Training plots: models/training_history.png")
    print()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
        sys.exit(130)

