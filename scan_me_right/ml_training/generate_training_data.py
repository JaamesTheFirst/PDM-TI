"""Training data generator for document formatting recognition.

This version builds synthetic multi-section document pages that include
headers, paragraphs, bold/italic/underlined emphasis, bullet lists, and
numbered lists. Each block is cropped into a standalone training sample
with formatting metadata so the downstream model can learn richer
formatting cues than single-line snippets.
"""

from __future__ import annotations

import json
import os
import random
import shutil
import textwrap
from datetime import datetime
from typing import Dict, List, Tuple
from typing import Set

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OUTPUT_DIR = "training_data"
PAGE_COUNT = 400  # Number of synthetic pages to generate
PAGE_SIZE = (1240, 1754)  # Approx A4 at ~150 DPI
BLOCK_IMAGE_SIZE = (224, 224)  # Model input size
PAGE_MARGIN = 80
BACKGROUND_COLORS = [
    (255, 255, 255),
    (250, 250, 250),
    (245, 245, 245),
]

BASE_SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning enables computers to adapt from data.",
    "Document security is critical for privacy-first applications.",
    "Building mobile apps requires attention to UX and performance.",
    "TensorFlow Lite brings deep learning to edge devices.",
    "Offline workflows give users full control of their information.",
    "Neural networks learn hierarchical representations automatically.",
    "Optical character recognition extracts text from images.",
    "Encryption protects sensitive content against unauthorized access.",
    "Unit testing ensures stable releases and quick regression detection.",
]

LIST_CANDIDATES = [
    "Review the captured scan for clarity before saving.",
    "Trim the document edges to reduce background noise.",
    "Confirm formatting hints like bold and italics were detected.",
    "Export the finalized document to PDF and TXT formats.",
    "Share securely via encrypted channels where possible.",
    "Double-check list numbering and bullet consistency.",
    "Ensure underlined emphasis is preserved in the output.",
    "Organize scans in folders by client or project name.",
    "Back up critical documents to offline media regularly.",
]

TITLE_TEXTS = [
    "Executive Summary",
    "Project Overview",
    "Key Findings",
    "Implementation Plan",
    "Design Specifications",
    "Meeting Minutes",
    "User Research Insights",
    "Release Notes",
]

BLOCK_TYPES = [
    "title",
    "paragraph",
    "bold",
    "italic",
    "underline",
    "bullet_list",
    "numbered_list",
]

STYLE_FLAGS = {
    "title": {"is_bold": True, "is_italic": False, "is_underlined": False, "is_list": False},
    "paragraph": {"is_bold": False, "is_italic": False, "is_underlined": False, "is_list": False},
    "bold": {"is_bold": True, "is_italic": False, "is_underlined": False, "is_list": False},
    "italic": {"is_bold": False, "is_italic": True, "is_underlined": False, "is_list": False},
    "underline": {"is_bold": False, "is_italic": False, "is_underlined": True, "is_list": False},
    "bullet_list": {"is_bold": False, "is_italic": False, "is_underlined": False, "is_list": True},
    "numbered_list": {"is_bold": False, "is_italic": False, "is_underlined": False, "is_list": True},
}

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _safe_crop_box(bbox: Tuple[int, int, int, int], image_size: Tuple[int, int], padding: int = 16) -> Tuple[int, int, int, int]:
    left = max(bbox[0] - padding, 0)
    top = max(bbox[1] - padding, 0)
    right = min(bbox[2] + padding, image_size[0])
    bottom = min(bbox[3] + padding, image_size[1])
    return left, top, right, bottom


def _wrap_list_item(text: str, prefix: str, width: int) -> List[str]:
    lines = textwrap.wrap(text, width=width)
    wrapped: List[str] = []
    for idx, line in enumerate(lines):
        wrapped.append((prefix if idx == 0 else " " * len(prefix)) + line)
    return wrapped


# ---------------------------------------------------------------------------
# Generator implementation
# ---------------------------------------------------------------------------


class DocumentDataGenerator:
    def __init__(self, output_dir: str = OUTPUT_DIR) -> None:
        self.output_dir = output_dir
        self.images_dir = os.path.join(output_dir, "images")
        self.labels_dir = os.path.join(output_dir, "labels")
        self.pages_dir = os.path.join(output_dir, "pages")

        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)

        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.labels_dir, exist_ok=True)
        os.makedirs(self.pages_dir, exist_ok=True)

        self.metadata: List[Dict] = []
        self.pages_metadata: List[Dict] = []
        self.sample_id = 0
        self.style_counts: Dict[str, int] = {}
        self.font_paths = self._resolve_font_paths()

    @staticmethod
    def _resolve_font_paths() -> Dict[str, str]:
        import platform

        system = platform.system()
        if system == "Linux":
            return {
                "normal": "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "bold": "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                "italic": "/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf",
            }
        if system == "Darwin":
            return {
                "normal": "/System/Library/Fonts/Helvetica.ttc",
                "bold": "/System/Library/Fonts/Helvetica.ttc",
                "italic": "/System/Library/Fonts/Helvetica.ttc",
            }
        return {
            "normal": "C:/Windows/Fonts/arial.ttf",
            "bold": "C:/Windows/Fonts/arialbd.ttf",
            "italic": "C:/Windows/Fonts/ariali.ttf",
        }

    def _load_font(self, style: str, size: int) -> ImageFont.FreeTypeFont:
        font_key = "normal"
        if style in {"bold", "title"}:
            font_key = "bold"
        elif style == "italic":
            font_key = "italic"

        font_path = self.font_paths.get(font_key)
        if font_path:
            try:
                return ImageFont.truetype(font_path, size)
            except OSError:
                pass
        # Fallbacks
        for candidate in self.font_paths.values():
            try:
                return ImageFont.truetype(candidate, size)
            except OSError:
                continue
        return ImageFont.load_default()

    def _augment_block(self, image: Image.Image) -> Image.Image:
        angle = random.uniform(-3, 3)
        image = image.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=(255, 255, 255))

        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(random.uniform(0.85, 1.15))

        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(random.uniform(0.9, 1.1))

        if random.random() < 0.25:
            image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.0)))

        if random.random() < 0.15:
            pixels = image.load()
            width, height = image.size
            for _ in range(int(width * height * 0.005)):
                x = random.randrange(width)
                y = random.randrange(height)
                noise = random.randint(-25, 25)
                r, g, b = pixels[x, y]
                pixels[x, y] = (
                    max(0, min(255, r + noise)),
                    max(0, min(255, g + noise)),
                    max(0, min(255, b + noise)),
                )
        return image

    @staticmethod
    def _build_paragraph() -> str:
        sentence_count = random.randint(2, 4)
        sentences = random.sample(BASE_SENTENCES, k=sentence_count)
        return " ".join(sentences)

    @staticmethod
    def _build_list(style: str) -> str:
        item_count = random.randint(3, 5)
        items = random.sample(LIST_CANDIDATES, k=item_count)
        lines: List[str] = []
        for idx, item in enumerate(items, start=1):
            prefix = "- " if style == "bullet_list" else f"{idx}. "
            lines.extend(_wrap_list_item(item, prefix=prefix, width=48))
        return "\n".join(lines)

    def _create_block_text(self, style: str) -> str:
        if style == "title":
            return random.choice(TITLE_TEXTS)
        if style in {"paragraph", "bold", "italic", "underline"}:
            return textwrap.fill(self._build_paragraph(), width=60)
        if style in {"bullet_list", "numbered_list"}:
            return self._build_list(style)
        raise ValueError(f"Unsupported style: {style}")

    def _draw_block(
        self,
        draw: ImageDraw.ImageDraw,
        image: Image.Image,
        style: str,
        y_cursor: int,
        page_index: int,
        features_present: Set[str],
    ) -> Tuple[int, int]:
        if style == "title":
            font_size = random.randint(46, 60)
            spacing = random.randint(8, 14)
        elif style in {"paragraph", "bold", "italic", "underline"}:
            font_size = random.randint(26, 32)
            spacing = random.randint(8, 12)
        else:  # lists
            font_size = random.randint(24, 30)
            spacing = random.randint(10, 14)

        font = self._load_font(style, font_size)
        block_text = self._create_block_text(style)
        text_position = (PAGE_MARGIN, y_cursor)

        bbox = draw.multiline_textbbox(text_position, block_text, font=font, spacing=spacing)
        block_height = bbox[3] - bbox[1]
        if bbox[3] + PAGE_MARGIN >= image.height:
            return y_cursor, 0

        draw.multiline_text(text_position, block_text, font=font, fill=(0, 0, 0), spacing=spacing)
        if style == "underline":
            underline_y = bbox[3] + 4
            draw.line([(bbox[0], underline_y), (bbox[2], underline_y)], fill=(0, 0, 0), width=3)
            bbox = (bbox[0], bbox[1], bbox[2], underline_y + 3)

        self._save_block_sample(
            image=image,
            bbox=bbox,
            style=style,
            text=block_text,
            font_size=font_size,
            line_spacing=spacing,
            page_index=page_index,
        )
        features_present.add(style)

        next_y = bbox[3] + random.randint(28, 60)
        return next_y, block_height

    def _save_block_sample(
        self,
        *,
        image: Image.Image,
        bbox: Tuple[int, int, int, int],
        style: str,
        text: str,
        font_size: int,
        line_spacing: int,
        page_index: int,
    ) -> None:
        crop_box = _safe_crop_box(bbox, image.size)
        block_image = image.crop(crop_box)
        block_image = self._augment_block(block_image)
        block_image = block_image.resize(BLOCK_IMAGE_SIZE, Image.LANCZOS)

        filename = f"{style}_{self.sample_id:06d}.jpg"
        filepath = os.path.join(self.images_dir, filename)
        block_image.save(filepath, quality=90)

        label = {
            "filename": filename,
            "text": text,
            "style": style,
            "page_index": page_index,
            "font_size": font_size,
            "line_spacing": line_spacing,
            "bounding_box": {
                "left": crop_box[0],
                "top": crop_box[1],
                "right": crop_box[2],
                "bottom": crop_box[3],
            },
        }
        label.update(STYLE_FLAGS[style])

        label_filename = f"{style}_{self.sample_id:06d}.json"
        with open(os.path.join(self.labels_dir, label_filename), "w", encoding="utf-8") as handle:
            json.dump(label, handle, indent=2)

        self.metadata.append(label)
        self.style_counts[style] = self.style_counts.get(style, 0) + 1
        self.sample_id += 1

    def generate_dataset(self, page_count: int = PAGE_COUNT) -> None:
        print("Starting training data generation...")
        print(f"Output directory: {self.output_dir}")
        print(f"Generating {page_count} synthetic pages with rich formatting.\n")

        for page_index in range(page_count):
            if (page_index + 1) % 25 == 0:
                print(f"  Rendering page {page_index + 1}/{page_count}")
            self._render_page(page_index)

        metadata_path = os.path.join(self.output_dir, "metadata.json")
        with open(metadata_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "generated_at": datetime.now().isoformat(),
                    "total_block_samples": len(self.metadata),
                    "total_pages": len(self.pages_metadata),
                    "blocks": self.metadata,
                    "pages": self.pages_metadata,
                },
                handle,
                indent=2,
            )

        print("\n✅ Dataset generation complete!")
        print(f"Total pages: {len(self.pages_metadata)}")
        print(f"Total block samples: {len(self.metadata)}")
        print(f"Page images: {self.pages_dir}")
        print(f"Block images: {self.images_dir}")
        print(f"Block labels: {self.labels_dir}")
        print(f"Metadata: {metadata_path}\n")

        print("Dataset statistics:")
        for style in sorted(self.style_counts):
            print(f"  {style:13s}: {self.style_counts[style]} samples")

    def _render_page(self, page_index: int) -> None:
        background_color = random.choice(BACKGROUND_COLORS)
        page = Image.new("RGB", PAGE_SIZE, background_color)
        draw = ImageDraw.Draw(page)

        y_cursor = PAGE_MARGIN
        block_plan = BLOCK_TYPES.copy()
        random.shuffle(block_plan)
        features_present: Set[str] = set()

        # Encourage title to appear near the top
        if "title" in block_plan:
            block_plan.remove("title")
            block_plan.insert(0, "title")

        for style in block_plan:
            y_cursor, block_height = self._draw_block(draw, page, style, y_cursor, page_index, features_present)
            if block_height == 0 or y_cursor >= PAGE_SIZE[1] - PAGE_MARGIN:
                break

        # Optional extra paragraphs near the end if space remains
        while y_cursor < PAGE_SIZE[1] - PAGE_MARGIN - 120:
            y_cursor, block_height = self._draw_block(draw, page, "paragraph", y_cursor, page_index, features_present)
            if block_height == 0:
                break

        self._save_page_image(page, page_index, features_present)

    def _save_page_image(self, page: Image.Image, page_index: int, features_present: Set[str]) -> None:
        page_copy = page.copy()

        enhancer = ImageEnhance.Brightness(page_copy)
        page_copy = enhancer.enhance(random.uniform(0.95, 1.05))

        enhancer = ImageEnhance.Contrast(page_copy)
        page_copy = enhancer.enhance(random.uniform(0.95, 1.05))

        if random.random() < 0.1:
            page_copy = page_copy.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))

        filename = f"page_{page_index:04d}.jpg"
        filepath = os.path.join(self.pages_dir, filename)
        page_copy.save(filepath, quality=90)

        feature_flags = {style: int(style in features_present) for style in STYLE_FLAGS.keys()}
        page_metadata = {
            "filename": filename,
            "features": sorted([style for style, present in feature_flags.items() if present]),
            "feature_flags": feature_flags,
        }
        self.pages_metadata.append(page_metadata)


def main() -> None:
    print("=" * 60)
    print("Document Formatting Training Data Generator")
    print("=" * 60)

    generator = DocumentDataGenerator()
    generator.generate_dataset()

    print("\n" + "=" * 60)
    print("Next steps:")
    print("1. Inspect generated page images in:", generator.pages_dir)
    print("2. Review block crops/labels in:", generator.images_dir)
    print("3. Train the formatting classifier with train_model.py")
    print("4. Export the best model to TensorFlow Lite")
    print("5. Ship the .tflite model with the Flutter app")
    print("=" * 60)


if __name__ == "__main__":
    main()

