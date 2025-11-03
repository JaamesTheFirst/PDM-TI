#!/usr/bin/env python3
"""
Full ML Training Pipeline - Real-time Output Version
Runs data generation and model training sequentially with unbuffered output
"""

import os
import sys
import subprocess

# Force unbuffered output
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(line_buffering=True)

def run_command(cmd, description):
    """Run a command with real-time output streaming"""
    print("=" * 60, flush=True)
    print(description, flush=True)
    print("=" * 60, flush=True)
    print(flush=True)
    
    try:
        # Use Popen for real-time streaming output
        # stdout and stderr go directly to parent's stdout/stderr
        process = subprocess.Popen(
            cmd,
            shell=True,
            stdout=sys.stdout,
            stderr=sys.stderr,
            text=True,
            bufsize=0  # Unbuffered for instant output
        )
        
        # Wait for completion
        return_code = process.wait()
        
        print(flush=True)
        
        if return_code != 0:
            print(f"\n❌ Error: {description} failed!", flush=True)
            print(f"Exit code: {return_code}", flush=True)
            sys.exit(1)
        
        return True
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user", flush=True)
        if 'process' in locals():
            process.terminate()
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}", flush=True)
        sys.exit(1)

def main():
    """Main pipeline execution"""
    print("=" * 60, flush=True)
    print("Document Formatting Recognition - Full Training Pipeline", flush=True)
    print("Real-time Output Version", flush=True)
    print("=" * 60, flush=True)
    print(flush=True)
    
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
    # Use -u flag for unbuffered output
    run_command(
        "python3 -u train_model.py",
        "Step 2/2: Training Model"
    )
    
    # Summary
    print(flush=True)
    print("=" * 60, flush=True)
    print("✅ Pipeline Complete!", flush=True)
    print("=" * 60, flush=True)
    print(flush=True)
    print("Results:", flush=True)
    print("  - Best model: models/best_model.keras", flush=True)
    print("  - TFLite model: models/formatting_classifier.tflite", flush=True)
    print("  - Training plots: models/training_history.png", flush=True)
    print(flush=True)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user", flush=True)
        sys.exit(130)

