"""Training data generator for document formatting recognition.

This version builds synthetic multi-section document pages that include
headers, paragraphs, bold/italic/underlined emphasis, bullet lists, and
numbered lists. Each block is cropped into a standalone training sample
with formatting metadata so the downstream model can learn richer
formatting cues than single-line snippets.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import textwrap
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from typing import Dict, List, Tuple, Set, Optional

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OUTPUT_DIR = "training_data"
TEST_OUTPUT_DIR = "test_images"
PAGE_COUNT = 1000  # Number of synthetic pages to generate
PAGE_SIZE = (1240, 1754)  # Approx A4 at ~150 DPI
BLOCK_IMAGE_SIZE = (224, 224)  # Model input size
PAGE_MARGIN = 80
SAMPLE_ID_STRIDE = 1_000_000  # ensures unique filenames per worker
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

QUOTE_ATTRIBUTIONS = [
    "A. Smith",
    "J. Rivera",
    "L. Thompson",
    "M. Patel",
    "S. Fernandez",
]

CAPTION_PREFIXES = ["Figure", "Table", "Exhibit"]
CALL_OUT_LABELS = ["Note", "Reminder", "Action Item", "Key Point"]
COMPANY_NAMES = [
    "Northwind Holdings",
    "Aurora Systems",
    "Summit Legal Group",
    "Blue Harbor Bank",
    "Vertex Consulting",
]
FOOTER_TAGLINES = [
    "Confidential",
    "Internal Use Only",
    "Draft Copy",
    "For Review",
]
SIGNATORY_NAMES = [
    "Alex Rivera",
    "Jamie Chen",
    "Morgan Patel",
    "Taylor Brooks",
    "Jordan Hughes",
]
SIGNATURE_TITLES = [
    "Chief Executive Officer",
    "Finance Director",
    "Project Lead",
    "Operations Manager",
    "Legal Counsel",
]
TABLE_HEADER_SETS = [
    ["Item", "Description", "Amount"],
    ["Date", "Owner", "Status"],
    ["Task", "Due", "Priority"],
    ["Dept", "Contact", "Phone"],
]

BLOCK_TYPES = [
    "title",
    "heading",
    "subheading",
    "paragraph",
    "quote",
    "callout",
    "bullet_list",
    "numbered_list",
    "table_header",
    "table_cell",
    "caption",
    "header",
    "footer",
    "page_number",
    "footnote",
    "signature",
    "sign_here",
    "line_break",
]

STYLE_CONFIG = {
    "title": {
        "font_key": "bold",
        "font_size": (46, 60),
        "line_spacing": (8, 14),
        "flags": {"is_bold": True, "is_italic": False, "is_underlined": False, "is_list": False},
    },
    "heading": {
        "font_key": "bold",
        "font_size": (34, 42),
        "line_spacing": (6, 10),
        "flags": {"is_bold": True, "is_italic": False, "is_underlined": False, "is_list": False},
    },
    "subheading": {
        "font_key": "italic",
        "font_size": (28, 34),
        "line_spacing": (6, 10),
        "flags": {"is_bold": False, "is_italic": True, "is_underlined": False, "is_list": False},
    },
    "paragraph": {
        "font_key": "normal",
        "font_size": (26, 32),
        "line_spacing": (8, 12),
        "flags": {"is_bold": False, "is_italic": False, "is_underlined": False, "is_list": False},
    },
    "quote": {
        "font_key": "italic",
        "font_size": (26, 32),
        "line_spacing": (10, 14),
        "flags": {"is_bold": False, "is_italic": True, "is_underlined": False, "is_list": False},
    },
    "callout": {
        "font_key": "bold",
        "font_size": (24, 30),
        "line_spacing": (10, 14),
        "flags": {"is_bold": True, "is_italic": False, "is_underlined": False, "is_list": False},
    },
    "bullet_list": {
        "font_key": "normal",
        "font_size": (24, 30),
        "line_spacing": (10, 14),
        "flags": {"is_bold": False, "is_italic": False, "is_underlined": False, "is_list": True},
    },
    "numbered_list": {
        "font_key": "normal",
        "font_size": (24, 30),
        "line_spacing": (10, 14),
        "flags": {"is_bold": False, "is_italic": False, "is_underlined": False, "is_list": True},
    },
    "table_header": {
        "font_key": "bold",
        "font_size": (22, 26),
        "line_spacing": (8, 10),
        "flags": {"is_bold": True, "is_italic": False, "is_underlined": False, "is_list": False},
    },
    "table_cell": {
        "font_key": "normal",
        "font_size": (22, 26),
        "line_spacing": (8, 10),
        "flags": {"is_bold": False, "is_italic": False, "is_underlined": False, "is_list": False},
    },
    "caption": {
        "font_key": "italic",
        "font_size": (20, 24),
        "line_spacing": (6, 8),
        "flags": {"is_bold": False, "is_italic": True, "is_underlined": False, "is_list": False},
    },
    "header": {
        "font_key": "normal",
        "font_size": (18, 22),
        "line_spacing": (4, 6),
        "flags": {"is_bold": False, "is_italic": False, "is_underlined": False, "is_list": False},
    },
    "footer": {
        "font_key": "normal",
        "font_size": (18, 22),
        "line_spacing": (4, 6),
        "flags": {"is_bold": False, "is_italic": False, "is_underlined": False, "is_list": False},
    },
    "page_number": {
        "font_key": "normal",
        "font_size": (18, 22),
        "line_spacing": (4, 6),
        "flags": {"is_bold": False, "is_italic": False, "is_underlined": False, "is_list": False},
    },
    "footnote": {
        "font_key": "normal",
        "font_size": (18, 22),
        "line_spacing": (6, 8),
        "flags": {"is_bold": False, "is_italic": False, "is_underlined": False, "is_list": False},
    },
    "signature": {
        "font_key": "italic",
        "font_size": (22, 28),
        "line_spacing": (8, 10),
        "flags": {"is_bold": False, "is_italic": True, "is_underlined": False, "is_list": False},
    },
    "sign_here": {
        "font_key": "normal",
        "font_size": (20, 24),
        "line_spacing": (8, 10),
        "flags": {"is_bold": False, "is_italic": False, "is_underlined": True, "is_list": False},
    },
    "line_break": {
        "font_key": "normal",
        "font_size": (14, 16),
        "line_spacing": (4, 6),
        "flags": {"is_bold": False, "is_italic": False, "is_underlined": False, "is_list": False},
    },
}

STYLE_FLAGS = {style: config["flags"] for style, config in STYLE_CONFIG.items()}

if set(STYLE_CONFIG.keys()) != set(BLOCK_TYPES):
    raise ValueError("STYLE_CONFIG keys must match BLOCK_TYPES")

FEATURE_PROBABILITIES = {
    "title": 0.8,
    "heading": 0.6,
    "subheading": 0.55,
    "paragraph": 0.95,
    "quote": 0.3,
    "callout": 0.25,
    "bullet_list": 0.45,
    "numbered_list": 0.4,
    "table_header": 0.2,
    "table_cell": 0.25,
    "caption": 0.25,
    "header": 0.35,
    "footer": 0.35,
    "page_number": 0.35,
    "footnote": 0.2,
    "signature": 0.2,
    "sign_here": 0.15,
    "line_break": 0.3,
}

MIN_BLOCKS_PER_PAGE = 2
MAX_BLOCKS_PER_PAGE = 6
TEMPLATE_PROBABILITY = 0.25

LAYOUT_TEMPLATES = [
    ["header", "title", "heading", "paragraph", "footer", "page_number"],
    ["title", "heading", "paragraph", "bullet_list", "footnote"],
    ["title", "paragraph", "quote", "caption"],
    ["title", "table_header", "table_cell", "table_cell", "caption"],
    ["title", "paragraph", "callout", "paragraph"],
    ["title", "paragraph", "numbered_list", "signature", "sign_here"],
    ["header", "title", "subheading", "paragraph", "bullet_list", "footer"],
    ["title", "subheading", "paragraph", "line_break", "paragraph", "footnote"],
    ["header", "title", "paragraph", "table_header", "table_cell", "table_cell", "footer"],
    ["title", "callout", "paragraph", "numbered_list", "page_number"],
    ["title", "paragraph", "quote", "line_break", "signature", "sign_here"],
    ["header", "paragraph", "table_header", "table_cell", "caption", "page_number"],
    ["title", "subheading", "paragraph", "bullet_list", "numbered_list", "footer"],
    ["title", "paragraph", "callout", "bullet_list", "footnote", "footer"],
]

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

# safe crop box to avoid out of bounds errors
def _safe_crop_box(bbox: Tuple[int, int, int, int], image_size: Tuple[int, int], padding: int = 16) -> Tuple[int, int, int, int]:
    left = max(bbox[0] - padding, 0)
    top = max(bbox[1] - padding, 0)
    right = min(bbox[2] + padding, image_size[0])
    bottom = min(bbox[3] + padding, image_size[1])
    return left, top, right, bottom

# wrap list item to avoid text wrapping errors
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
    def __init__(
        self,
        output_dir: str = OUTPUT_DIR,
        *,
        clean_output: bool = True,
        sample_id_offset: int = 0,
        page_offset: int = 0,
        worker_id: int = 0,
        metadata_filename: Optional[str] = None,
        quiet: bool = False,
    ) -> None:
        self.output_dir = output_dir
        self.images_dir = os.path.join(output_dir, "images")
        self.labels_dir = os.path.join(output_dir, "labels")
        self.pages_dir = os.path.join(output_dir, "pages")
        self.worker_id = worker_id
        self.page_offset = page_offset
        self.sample_id = sample_id_offset
        self.metadata_filename = metadata_filename or "metadata.json"
        self.metadata_output_path = os.path.join(self.output_dir, self.metadata_filename)
        self.quiet = quiet

        if clean_output and os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)

        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.labels_dir, exist_ok=True)
        os.makedirs(self.pages_dir, exist_ok=True)

        self.metadata: List[Dict] = []
        self.pages_metadata: List[Dict] = []
        self.style_counts: Dict[str, int] = {}
        self.font_paths = self._resolve_font_paths()

    def _log(self, message: str) -> None:
        if not self.quiet:
            print(message)

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

    # load font to avoid font loading errors first
    def _load_font(self, font_key: str, size: int) -> ImageFont.FreeTypeFont:
        font_path = self.font_paths.get(font_key, self.font_paths.get("normal"))
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

    
    #randomizes contrast, brightness, and rotation between 
    def _augment_block(self, image: Image.Image) -> Image.Image:
        angle = random.uniform(-3, 3)
        image = image.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=(255, 255, 255))

        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(random.uniform(0.85, 1.15))

        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(random.uniform(0.9, 1.1))

        # randomly blur the image
        if random.random() < 0.25:
            image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.0)))

        # randomly add noise to the image
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

    #build paragraph 
    @staticmethod
    def _build_paragraph() -> str:
        sentence_count = random.randint(2, 4)
        sentences = random.sample(BASE_SENTENCES, k=sentence_count)
        return " ".join(sentences)

    #builds list of items for bullet and numbered lists
    @staticmethod
    def _build_list(style: str) -> str:
        item_count = random.randint(3, 5)
        items = random.sample(LIST_CANDIDATES, k=item_count)
        lines: List[str] = []
        for idx, item in enumerate(items, start=1):
            prefix = "- " if style == "bullet_list" else f"{idx}. "
            lines.extend(_wrap_list_item(item, prefix=prefix, width=48))
        return "\n".join(lines)

    #defines a block of text for a given style
    def _create_block_text(self, style: str, page_index: int) -> str:
        if style == "title":
            return random.choice(TITLE_TEXTS)
        if style == "heading":
            return random.choice(TITLE_TEXTS)
        if style == "subheading":
            text = f"{random.choice(TITLE_TEXTS)} Details"
            return textwrap.fill(text, width=54)
        if style == "paragraph":
            return textwrap.fill(self._build_paragraph(), width=60)
        if style == "quote":
            return self._build_quote()
        if style == "callout":
            return self._build_callout()
        if style in {"bullet_list", "numbered_list"}:
            return self._build_list(style)
        if style == "table_header":
            return self._build_table_block(include_header=True)
        if style == "table_cell":
            return self._build_table_block(include_header=False)
        if style == "caption":
            return self._build_caption()
        if style == "header":
            return self._build_header_text(page_index)
        if style == "footer":
            return self._build_footer_text(page_index)
        if style == "page_number":
            return self._build_page_number_text(page_index)
        if style == "footnote":
            return self._build_footnote_text()
        if style == "signature":
            return self._build_signature_text()
        if style == "sign_here":
            return self._build_sign_here_text()
        if style == "line_break":
            return self._build_line_break()
        raise ValueError(f"Unsupported style: {style}")

    def _build_quote(self) -> str:
        body = textwrap.fill(self._build_paragraph(), width=58)
        author = random.choice(QUOTE_ATTRIBUTIONS)
        return f"\"{body}\"\n- {author}"

    def _build_callout(self) -> str:
        label = random.choice(CALL_OUT_LABELS).upper()
        body = textwrap.fill(self._build_paragraph(), width=52)
        return f"{label}\n{body}"

    def _build_table_block(self, *, include_header: bool) -> str:
        columns = random.choice(TABLE_HEADER_SETS)
        if include_header:
            header_line = " | ".join(column.upper() for column in columns)
            separator = "-+-".join("-" * max(len(column), 4) for column in columns)
            return f"{header_line}\n{separator}"

        rows = []
        for _ in range(random.randint(2, 3)):
            cells = []
            for _column in columns:
                words = random.choice(BASE_SENTENCES).split()
                cells.append(" ".join(words[: random.randint(1, 3)]))
            rows.append(" | ".join(cells))
        return "\n".join(rows)

    def _build_caption(self) -> str:
        prefix = random.choice(CAPTION_PREFIXES)
        number = random.randint(1, 15)
        text = textwrap.fill(random.choice(BASE_SENTENCES), width=60)
        return f"{prefix} {number}. {text}"

    def _build_header_text(self, page_index: int) -> str:
        company = random.choice(COMPANY_NAMES)
        doc = random.choice(TITLE_TEXTS)
        return f"{company} | {doc} | Page {page_index + 1}"

    def _build_footer_text(self, page_index: int) -> str:
        tagline = random.choice(FOOTER_TAGLINES)
        year = random.randint(2018, datetime.now().year)
        return f"{tagline} - Page {page_index + 1} - {year}"

    def _build_page_number_text(self, page_index: int) -> str:
        total_pages = max(page_index + 1, random.randint(page_index + 1, page_index + 5))
        return f"Page {page_index + 1} of {total_pages}"

    def _build_footnote_text(self) -> str:
        marker = random.randint(1, 5)
        body = textwrap.fill(random.choice(BASE_SENTENCES), width=60)
        return f"{marker}. {body}"

    def _build_signature_text(self) -> str:
        line = "_" * random.randint(24, 34)
        name = random.choice(SIGNATORY_NAMES)
        title = random.choice(SIGNATURE_TITLES)
        return f"{line}\n{name}\n{title}"

    @staticmethod
    def _build_sign_here_text() -> str:
        return "Sign here: " + "_" * 28

    @staticmethod
    def _build_line_break() -> str:
        return "=" * 48

    #draws a block of text on a page
    def _draw_block(
        self,
        draw: ImageDraw.ImageDraw,
        image: Image.Image,
        style: str,
        y_cursor: int,
        page_index: int,
        features_present: Set[str],
    ) -> Tuple[int, int]:
        config = STYLE_CONFIG.get(style)
        if config is None:
            raise ValueError(f"Unknown style requested: {style}")

        font_size = random.randint(*config["font_size"])
        spacing = random.randint(*config["line_spacing"])
        font = self._load_font(config["font_key"], font_size)
        block_text = self._create_block_text(style, page_index=page_index)
        text_position = (PAGE_MARGIN, y_cursor)

        bbox = draw.multiline_textbbox(text_position, block_text, font=font, spacing=spacing)
        block_height = bbox[3] - bbox[1]
        if bbox[3] + PAGE_MARGIN >= image.height:
            return y_cursor, 0

        draw.multiline_text(text_position, block_text, font=font, fill=(0, 0, 0), spacing=spacing)

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

    #saves a block of text as an image and label
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

    #generates a dataset of pages with blocks of text (as many as PAGE_COUNT defines)
    def generate_dataset(self, page_count: int = PAGE_COUNT) -> None:
        self._log("Starting training data generation...")
        self._log(f"Output directory: {self.output_dir}")
        self._log(f"Generating {page_count} synthetic pages with rich formatting.\n")

        for page_index in range(page_count):
            if not self.quiet and (page_index + 1) % 25 == 0:
                self._log(f"  Rendering page {page_index + 1}/{page_count}")
            self._render_page(self.page_offset + page_index)

        with open(self.metadata_output_path, "w", encoding="utf-8") as handle:
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

        self._log("\n✅ Dataset generation complete!")
        self._log(f"Total pages: {len(self.pages_metadata)}")
        self._log(f"Total block samples: {len(self.metadata)}")
        self._log(f"Page images: {self.pages_dir}")
        self._log(f"Block images: {self.images_dir}")
        self._log(f"Block labels: {self.labels_dir}")
        self._log(f"Metadata: {self.metadata_output_path}\n")

        if not self.quiet:
            self._log("Dataset statistics:")
            for style in sorted(self.style_counts):
                self._log(f"  {style:13s}: {self.style_counts[style]} samples")

    def _render_page(self, page_index: int) -> None:
        background_color = random.choice(BACKGROUND_COLORS)
        page = Image.new("RGB", PAGE_SIZE, background_color)
        draw = ImageDraw.Draw(page)

        y_cursor = PAGE_MARGIN
        block_plan = self._choose_block_plan()
        features_present: Set[str] = set()

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

    def _choose_block_plan(self) -> List[str]:
        if random.random() < TEMPLATE_PROBABILITY:
            plan = random.choice(LAYOUT_TEMPLATES).copy()
        else:
            candidates = [
                style
                for style in BLOCK_TYPES
                if random.random() < FEATURE_PROBABILITIES.get(style, 0.5)
            ]
            if not candidates:
                candidates = ["paragraph"]
            random.shuffle(candidates)
            limit = random.randint(MIN_BLOCKS_PER_PAGE, max(MIN_BLOCKS_PER_PAGE, min(MAX_BLOCKS_PER_PAGE, len(candidates))))
            plan = candidates[:limit]

        if "paragraph" not in plan:
            plan.insert(1 if plan else 0, "paragraph")

        if plan[0] != "title" and "title" in plan:
            plan.remove("title")
            plan.insert(0, "title")

        return plan

    def _save_page_image(self, page: Image.Image, page_index: int, features_present: Set[str]) -> None:
        page_copy = page.copy()

        enhancer = ImageEnhance.Brightness(page_copy)
        page_copy = enhancer.enhance(random.uniform(0.95, 1.05))

        enhancer = ImageEnhance.Contrast(page_copy)
        page_copy = enhancer.enhance(random.uniform(0.95, 1.05))

        if random.random() < 0.1:
            page_copy = page_copy.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))

        filename = f"page_{page_index:06d}.jpg"
        filepath = os.path.join(self.pages_dir, filename)
        page_copy.save(filepath, quality=90)

        feature_flags = {style: int(style in features_present) for style in STYLE_FLAGS.keys()}
        page_metadata = {
            "filename": filename,
            "features": sorted([style for style, present in feature_flags.items() if present]),
            "feature_flags": feature_flags,
        }
        self.pages_metadata.append(page_metadata)


def _plan_worker_jobs(total_pages: int, workers: int) -> List[Dict[str, int]]:
    pages_per_worker = math.ceil(total_pages / workers)
    tasks: List[Dict[str, int]] = []
    page_start = 0

    for worker_id in range(workers):
        if page_start >= total_pages:
            break
        count = min(pages_per_worker, total_pages - page_start)
        tasks.append(
            {
                "worker_id": worker_id,
                "page_offset": page_start,
                "page_count": count,
                "sample_offset": worker_id * SAMPLE_ID_STRIDE,
            }
        )
        page_start += count

    return tasks


def _run_generation_worker(params: Dict) -> Dict:
    generator = DocumentDataGenerator(
        output_dir=params["output_dir"],
        clean_output=params.get("clean_output", False),
        sample_id_offset=params["sample_offset"],
        page_offset=params["page_offset"],
        worker_id=params["worker_id"],
        metadata_filename=params["metadata_filename"],
        quiet=params.get("quiet", True),
    )
    generator.generate_dataset(page_count=params["page_count"])
    return {
        "metadata_path": generator.metadata_output_path,
        "style_counts": generator.style_counts,
        "total_pages": len(generator.pages_metadata),
        "total_blocks": len(generator.metadata),
        "worker_id": params["worker_id"],
    }


def _merge_metadata_files(metadata_files: List[str], output_path: str) -> Dict:
    combined_blocks: List[Dict] = []
    combined_pages: List[Dict] = []

    for path in sorted(metadata_files):
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        combined_blocks.extend(data.get("blocks", []))
        combined_pages.extend(data.get("pages", []))

    combined_blocks.sort(key=lambda item: item.get("filename", ""))
    combined_pages.sort(key=lambda item: item.get("filename", ""))

    merged = {
        "generated_at": datetime.now().isoformat(),
        "total_block_samples": len(combined_blocks),
        "total_pages": len(combined_pages),
        "blocks": combined_blocks,
        "pages": combined_pages,
    }

    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(merged, handle, indent=2)

    return merged


def _run_parallel_generation(page_count: int, output_dir: str, workers: int) -> None:
    tasks = _plan_worker_jobs(page_count, workers)
    if not tasks:
        print("No pages scheduled for generation.")
        return

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    worker_params = []
    for task in tasks:
        worker_params.append(
            {
                "output_dir": output_dir,
                "page_count": task["page_count"],
                "page_offset": task["page_offset"],
                "sample_offset": task["sample_offset"],
                "worker_id": task["worker_id"],
                "metadata_filename": f"metadata_worker_{task['worker_id']:02d}.json",
                "clean_output": False,
                "quiet": True,
            }
        )

    print(f"Spawning {len(worker_params)} workers (total pages: {page_count})...")
    worker_results = []
    with ProcessPoolExecutor(max_workers=len(worker_params)) as executor:
        future_map = {executor.submit(_run_generation_worker, params): params for params in worker_params}
        for future in as_completed(future_map):
            result = future.result()
            worker_results.append(result)
            print(
                f"  Worker {result['worker_id']:02d} finished "
                f"({result['total_pages']} pages, {result['total_blocks']} blocks)"
            )

    metadata_files = [res["metadata_path"] for res in worker_results]
    merged_metadata = _merge_metadata_files(metadata_files, os.path.join(output_dir, "metadata.json"))
    for path in metadata_files:
        try:
            os.remove(path)
        except OSError:
            pass

    total_pages = sum(res["total_pages"] for res in worker_results)
    total_blocks = sum(res["total_blocks"] for res in worker_results)
    style_counter: Counter = Counter()
    for res in worker_results:
        style_counter.update(res["style_counts"])

    print("\n✅ Parallel dataset generation complete!")
    print(f"Total pages: {total_pages}")
    print(f"Total block samples: {total_blocks}")
    print(f"Metadata: {os.path.join(output_dir, 'metadata.json')}")
    print(f"Blocks stored: {len(merged_metadata.get('blocks', []))}, Pages stored: {len(merged_metadata.get('pages', []))}\n")

    print("Dataset statistics:")
    for style, count in sorted(style_counter.items()):
        print(f"  {style:13s}: {count} samples")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synthetic document formatting data generator.")
    parser.add_argument(
        "--page-count",
        type=int,
        default=PAGE_COUNT,
        help="Number of synthetic pages to render.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel worker processes to use.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=OUTPUT_DIR,
        help="Destination directory for generated dataset.",
    )
    parser.add_argument(
        "--use-test-output",
        action="store_true",
        help=f"Shortcut to write outputs into '{TEST_OUTPUT_DIR}'.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_dir = TEST_OUTPUT_DIR if args.use_test_output else args.output_dir

    print("=" * 60)
    print("Document Formatting Training Data Generator")
    print("=" * 60)

    worker_count = max(1, args.workers)

    if worker_count > 1:
        _run_parallel_generation(args.page_count, output_dir, worker_count)
    else:
        generator = DocumentDataGenerator(output_dir=output_dir)
        generator.generate_dataset(page_count=args.page_count)

    print("\n" + "=" * 60)
    print("Next steps:")
    pages_dir = os.path.join(output_dir, "pages")
    images_dir = os.path.join(output_dir, "images")
    labels_dir = os.path.join(output_dir, "labels")
    metadata_path = os.path.join(output_dir, "metadata.json")
    print("1. Inspect generated page images in:", pages_dir)
    print("2. Review block crops/labels in:", images_dir)
    print("3. Train the formatting classifier with train_model.py")
    print("4. Export the best model to TensorFlow Lite")
    print("5. Ship the .tflite model with the Flutter app")
    print("\nTips:")
    print(f"- Use --page-count N for quick spot checks (e.g., N=10).")
    print(f"- Use --use-test-output to target '{TEST_OUTPUT_DIR}' when validating layouts.")
    print("- Override --output-dir for custom destinations.")
    print("- Pass --workers <N> to fan out generation across CPU cores.")
    print(f"- Metadata written to: {metadata_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()

