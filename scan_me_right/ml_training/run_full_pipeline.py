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
    sys.stdout.flush()  # Force flush output
    
    try:
        # Use Popen for real-time streaming output
        process = subprocess.Popen(
            cmd,
            shell=True,
            stdout=sys.stdout,
            stderr=sys.stderr,
            text=True,
            bufsize=1  # Line buffered
        )
        process.wait()  # Wait for completion
        
        if process.returncode != 0:
            print(f"\n❌ Error: {description} failed!")
            print(f"Exit code: {process.returncode}")
            sys.exit(1)
        
        print()
        sys.stdout.flush()
        return True
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
        if 'process' in locals():
            process.terminate()
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
    
    # Step 1: Generate Training Data (always runs, overwrites if exists)
    # Use -u flag for unbuffered output
    run_command(
        "python3 -u generate_training_data.py",
        "Step 1/2: Generating Training Data"
    )
    
    # Step 2: Train Model (loads all available data dynamically)
    run_command(
        "python3 -u train_model.py",
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



