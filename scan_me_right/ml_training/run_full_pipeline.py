#!/usr/bin/env python3
"""Full ML training pipeline with environment-agnostic configuration."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence


def run_command(cmd: Sequence[str], description: str, *, cwd: Path) -> None:
    """Run a command with streaming output and fail fast on errors."""
    print("=" * 60)
    print(description)
    print("=" * 60)
    print()
    sys.stdout.flush()

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")

    try:
        process = subprocess.Popen(  # noqa: S603,S607 - controlled input
            list(cmd),
            cwd=str(cwd),
            stdout=sys.stdout,
            stderr=sys.stderr,
            text=True,
            bufsize=1,
            env=env,
        )
        return_code = process.wait()

        if return_code != 0:
            print(f"\n❌ Error: {description} failed!")
            print(f"Exit code: {return_code}")
            sys.exit(return_code)

        print()
        sys.stdout.flush()
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
        if "process" in locals():
            process.terminate()
        sys.exit(130)
    except Exception as exc:  # pragma: no cover - best-effort fallback
        print(f"\n❌ Unexpected error while running '{description}': {exc}")
        sys.exit(1)


def main() -> None:
    """Main entry point for the full pipeline."""
    script_dir = Path(__file__).resolve().parent

    print("=" * 60)
    print("Document Formatting Recognition - Full Training Pipeline")
    print(f"Working directory: {script_dir}")
    print(f"Python interpreter: {sys.executable}")
    print("=" * 60)
    print()

    requirements_path = script_dir / "requirements.txt"
    if requirements_path.exists():
        run_command(
            [sys.executable, "-m", "pip", "install", "-r", str(requirements_path)],
            "Preparing environment: installing Python dependencies",
            cwd=script_dir,
        )
    else:
        print("⚠️  No requirements.txt found; skipping dependency installation.\n")

    run_command(
        [sys.executable, "-u", str(script_dir / "generate_training_data.py")],
        "Step 1/2: Generating Training Data",
        cwd=script_dir,
    )

    run_command(
        [sys.executable, "-u", str(script_dir / "train_model.py")],
        "Step 2/2: Training Model",
        cwd=script_dir,
    )

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
