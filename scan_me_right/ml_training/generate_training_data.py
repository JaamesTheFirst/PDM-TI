"""
Training Data Generation Script for Document Formatting Recognition
This script generates synthetic document images with various text styles
for training a TensorFlow Lite model to detect formatting (bold, italic, underline, headers)
"""

import os
import random
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import json
from datetime import datetime

# Configuration
OUTPUT_DIR = "training_data"
SAMPLES_PER_STYLE = 1000  # Adjust based on your needs
IMAGE_SIZE = (800, 600)
BACKGROUND_COLORS = [(255, 255, 255), (250, 250, 250), (245, 245, 245)]

# Sample texts
SAMPLE_TEXTS = [
    "The quick brown fox jumps over the lazy dog",
    "Machine Learning and Artificial Intelligence",
    "Document Scanner Application",
    "Privacy-First Technology",
    "Mobile Application Development",
    "Neural Networks and Deep Learning",
    "Text Recognition and OCR",
    "Software Engineering Best Practices",
]

TITLE_TEXTS = [
    "Chapter 1: Introduction",
    "Executive Summary",
    "Table of Contents",
    "Abstract",
    "Conclusion",
    "References",
]


class DocumentDataGenerator:
    def __init__(self, output_dir=OUTPUT_DIR):
        self.output_dir = output_dir
        self.images_dir = os.path.join(output_dir, "images")
        self.labels_dir = os.path.join(output_dir, "labels")
        
        # Create directories
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.labels_dir, exist_ok=True)
        
        # Metadata for all samples
        self.metadata = []
    
    def get_font_path(self):
        """
        Try to locate system fonts for different text styles
        Supports Linux (Azure), macOS, and Windows
        """
        import platform
        system = platform.system()
        
        if system == 'Linux':
            # Common Linux font paths (Azure/Linux)
            font_paths = {
                'normal': '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
                'bold': '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
                'italic': '/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf',
            }
            # Try to find available fonts
            fallback_fonts = [
                '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf',
                '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
            ]
        elif system == 'Darwin':  # macOS
            font_paths = {
                'normal': '/System/Library/Fonts/Helvetica.ttc',
                'bold': '/System/Library/Fonts/Helvetica.ttc',
                'italic': '/System/Library/Fonts/Helvetica.ttc',
            }
            fallback_fonts = ['/System/Library/Fonts/Helvetica.ttc']
        else:  # Windows
            font_paths = {
                'normal': 'C:/Windows/Fonts/arial.ttf',
                'bold': 'C:/Windows/Fonts/arialbd.ttf',
                'italic': 'C:/Windows/Fonts/ariali.ttf',
            }
            fallback_fonts = ['C:/Windows/Fonts/arial.ttf']
        
        return font_paths
    
    def apply_augmentation(self, image):
        """Apply realistic augmentations to simulate phone camera captures"""
        # Random rotation (-5 to 5 degrees)
        angle = random.uniform(-5, 5)
        image = image.rotate(angle, expand=True, fillcolor=(255, 255, 255))
        
        # Random brightness
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(random.uniform(0.8, 1.2))
        
        # Random contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(random.uniform(0.9, 1.1))
        
        # Slight blur
        if random.random() < 0.3:
            image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
        
        # Add noise
        if random.random() < 0.2:
            pixels = image.load()
            for i in range(image.size[0]):
                for j in range(image.size[1]):
                    if random.random() < 0.01:
                        noise = random.randint(-20, 20)
                        r, g, b = pixels[i, j]
                        pixels[i, j] = (
                            max(0, min(255, r + noise)),
                            max(0, min(255, g + noise)),
                            max(0, min(255, b + noise))
                        )
        
        return image
    
    def generate_sample(self, text, style, font_size, sample_id):
        """Generate a single training sample"""
        # Create image with random background
        bg_color = random.choice(BACKGROUND_COLORS)
        image = Image.new('RGB', IMAGE_SIZE, bg_color)
        draw = ImageDraw.Draw(image)
        
        # Try to load font (fallback to default if not available)
        font_paths = self.get_font_path()
        font = None
        
        # Try style-specific font first
        font_key = 'bold' if style == 'bold' else ('italic' if style == 'italic' else 'normal')
        if font_key in font_paths:
            try:
                font = ImageFont.truetype(font_paths[font_key], font_size)
            except:
                pass
        
        # Fallback to any available font
        if font is None:
            for font_path in font_paths.values():
                try:
                    font = ImageFont.truetype(font_path, font_size)
                    break
                except:
                    continue
        
        # Last resort: default font
        if font is None:
            font = ImageFont.load_default()
        
        # Random position
        x = random.randint(50, 150)
        y = random.randint(50, IMAGE_SIZE[1] - 200)
        
        # Draw text
        text_color = (0, 0, 0)
        draw.text((x, y), text, fill=text_color, font=font)
        
        # For underlined text, draw a line
        if style == 'underline':
            bbox = draw.textbbox((x, y), text, font=font)
            draw.line([(bbox[0], bbox[3] + 2), (bbox[2], bbox[3] + 2)], 
                     fill=text_color, width=2)
        
        # Apply augmentations
        image = self.apply_augmentation(image)
        
        # Save image
        image_filename = f"{style}_{sample_id:06d}.jpg"
        image_path = os.path.join(self.images_dir, image_filename)
        image.save(image_path, quality=85)
        
        # Create label
        label = {
            'filename': image_filename,
            'text': text,
            'style': style,
            'is_bold': style == 'bold',
            'is_italic': style == 'italic',
            'is_underlined': style == 'underline',
            'is_title': style == 'title',
            'font_size': font_size,
        }
        
        # Save label
        label_filename = f"{style}_{sample_id:06d}.json"
        label_path = os.path.join(self.labels_dir, label_filename)
        with open(label_path, 'w') as f:
            json.dump(label, f, indent=2)
        
        self.metadata.append(label)
        
        return image_path, label_path
    
    def generate_dataset(self):
        """Generate complete dataset"""
        print("Starting training data generation...")
        print(f"Output directory: {self.output_dir}")
        
        sample_id = 0
        
        # Generate normal text samples
        print("\nGenerating normal text samples...")
        for i in range(SAMPLES_PER_STYLE):
            text = random.choice(SAMPLE_TEXTS)
            self.generate_sample(text, 'normal', random.randint(12, 16), sample_id)
            sample_id += 1
            if (i + 1) % 100 == 0:
                print(f"  Generated {i + 1}/{SAMPLES_PER_STYLE}")
        
        # Generate bold text samples
        print("\nGenerating bold text samples...")
        for i in range(SAMPLES_PER_STYLE):
            text = random.choice(SAMPLE_TEXTS)
            self.generate_sample(text, 'bold', random.randint(12, 16), sample_id)
            sample_id += 1
            if (i + 1) % 100 == 0:
                print(f"  Generated {i + 1}/{SAMPLES_PER_STYLE}")
        
        # Generate italic text samples
        print("\nGenerating italic text samples...")
        for i in range(SAMPLES_PER_STYLE):
            text = random.choice(SAMPLE_TEXTS)
            self.generate_sample(text, 'italic', random.randint(12, 16), sample_id)
            sample_id += 1
            if (i + 1) % 100 == 0:
                print(f"  Generated {i + 1}/{SAMPLES_PER_STYLE}")
        
        # Generate underlined text samples
        print("\nGenerating underlined text samples...")
        for i in range(SAMPLES_PER_STYLE):
            text = random.choice(SAMPLE_TEXTS)
            self.generate_sample(text, 'underline', random.randint(12, 16), sample_id)
            sample_id += 1
            if (i + 1) % 100 == 0:
                print(f"  Generated {i + 1}/{SAMPLES_PER_STYLE}")
        
        # Generate title samples
        print("\nGenerating title samples...")
        for i in range(SAMPLES_PER_STYLE):
            text = random.choice(TITLE_TEXTS)
            self.generate_sample(text, 'title', random.randint(18, 28), sample_id)
            sample_id += 1
            if (i + 1) % 100 == 0:
                print(f"  Generated {i + 1}/{SAMPLES_PER_STYLE}")
        
        # Save metadata
        metadata_path = os.path.join(self.output_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump({
                'generated_at': datetime.now().isoformat(),
                'total_samples': len(self.metadata),
                'samples': self.metadata,
            }, f, indent=2)
        
        print(f"\n✅ Dataset generation complete!")
        print(f"Total samples: {len(self.metadata)}")
        print(f"Images: {self.images_dir}")
        print(f"Labels: {self.labels_dir}")
        print(f"Metadata: {metadata_path}")
        
        # Print dataset statistics
        print("\nDataset statistics:")
        styles = {}
        for sample in self.metadata:
            style = sample['style']
            styles[style] = styles.get(style, 0) + 1
        
        for style, count in sorted(styles.items()):
            print(f"  {style}: {count} samples")


def main():
    """Main function to run the generator"""
    print("=" * 60)
    print("Document Formatting Training Data Generator")
    print("=" * 60)
    
    generator = DocumentDataGenerator()
    generator.generate_dataset()
    
    print("\n" + "=" * 60)
    print("Next steps:")
    print("1. Review generated samples in:", generator.images_dir)
    print("2. Train a CNN/MobileNet model using this data")
    print("3. Export model to TensorFlow Lite format")
    print("4. Place .tflite model in assets/models/")
    print("5. Update OCR service to use ML model instead of heuristics")
    print("=" * 60)


if __name__ == "__main__":
    main()

