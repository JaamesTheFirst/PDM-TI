"""
Comprehensive Training Data Generator - Realistic Documents
Generates documents with ALL formatting types found in real documents:
- Titles, Headers (H1, H2, H3)
- Paragraphs
- Bold, Italic, Underline, Bold+Italic
- Bullet points, Numbered lists
- Horizontal lines
- Page breaks
- Blockquotes
- Indentation
- Mixed formatting on same line
"""

import os
import random
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import json
from datetime import datetime

# Configuration
OUTPUT_DIR = "training_data_comprehensive"
NUM_DOCUMENTS = 5000  # Start with 5k pages (can scale to 50k later)
IMAGE_SIZE = (1200, 1800)  # Realistic A4-ish page
MARGIN_LEFT = 80
MARGIN_RIGHT = 80
MARGIN_TOP = 100
MARGIN_BOTTOM = 100
LINE_SPACING = 1.5

# Font configuration for macOS
FONT_BASE_PATH = '/System/Library/Fonts/Helvetica.ttc'

# Content pools
DOCUMENT_TITLES = [
    "Research Methodology and Analysis",
    "Privacy-Preserving Technologies",
    "Machine Learning Applications",
    "Software Architecture Patterns",
    "Mobile Computing Systems",
    "Data Security Principles",
    "Artificial Intelligence Ethics",
    "Computer Vision Techniques",
]

SECTION_HEADERS = [
    "Introduction",
    "Background and Related Work",
    "Methodology",
    "Results and Discussion",
    "Implementation Details",
    "Performance Analysis",
    "Conclusion and Future Work",
    "References",
]

SUBSECTION_HEADERS = [
    "Data Collection",
    "Experimental Setup",
    "Key Findings",
    "Limitations",
    "Recommendations",
]

PARAGRAPHS = [
    "The rapid advancement of machine learning technologies has transformed numerous industries. "
    "Modern applications leverage sophisticated algorithms to extract meaningful patterns from vast datasets. "
    "This paradigm shift enables automated decision-making processes that were previously impossible.",
    
    "Privacy-first design principles ensure that sensitive user data remains protected throughout processing. "
    "By implementing local computation strategies, applications can minimize exposure to external threats. "
    "This approach is particularly critical for handling confidential documents and personal information.",
    
    "Mobile devices now possess sufficient computational power to execute complex neural networks. "
    "Frameworks such as TensorFlow Lite enable efficient deployment of machine learning models on smartphones. "
    "This capability facilitates offline processing and reduces dependency on cloud infrastructure.",
    
    "Optical character recognition systems have achieved remarkable accuracy improvements in recent years. "
    "Contemporary OCR technologies can process printed text with accuracy rates exceeding ninety-five percent. "
    "The primary challenge involves preserving document structure and formatting during extraction.",
    
    "Transfer learning methodologies allow practitioners to leverage pre-trained models for specialized tasks. "
    "By initializing with ImageNet weights, researchers can achieve high performance with limited training data. "
    "This technique proves especially valuable for resource-constrained mobile deployment scenarios.",
]

BULLET_ITEMS = [
    "Data preprocessing and cleaning procedures",
    "Model architecture selection criteria",
    "Training optimization strategies",
    "Validation and testing protocols",
    "Performance benchmarking methods",
    "Error analysis and debugging techniques",
    "Deployment and integration steps",
]

BOLD_INLINE = [
    "important consideration",
    "critical factor",
    "key finding",
    "significant result",
    "primary objective",
    "essential requirement",
]

ITALIC_INLINE = [
    "as previously mentioned",
    "according to recent studies",
    "in particular",
    "for instance",
    "it should be noted that",
]

BLOCKQUOTES = [
    "The only way to do great work is to love what you do.",
    "Innovation distinguishes between a leader and a follower.",
    "Quality is not an act, it is a habit.",
]


class ComprehensiveDocumentGenerator:
    def __init__(self, output_dir=OUTPUT_DIR):
        self.output_dir = output_dir
        self.images_dir = os.path.join(output_dir, "images")
        self.labels_dir = os.path.join(output_dir, "labels")
        
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.labels_dir, exist_ok=True)
        
        self.metadata = []
        self.current_y = 0
        self.max_width = IMAGE_SIZE[0] - MARGIN_LEFT - MARGIN_RIGHT
    
    def get_font(self, size):
        """Load font"""
        try:
            return ImageFont.truetype(FONT_BASE_PATH, size)
        except:
            return ImageFont.load_default()
    
    def apply_augmentation(self, image):
        """Realistic camera capture augmentations"""
        # Rotation (document not perfectly straight)
        angle = random.uniform(-4, 4)
        image = image.rotate(angle, expand=False, fillcolor=(255, 255, 255))
        
        # Lighting variations
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(random.uniform(0.80, 1.20))
        
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(random.uniform(0.85, 1.15))
        
        # Camera blur
        if random.random() < 0.4:
            image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 1.2)))
        
        # Shadows/uneven lighting
        if random.random() < 0.3:
            # Add subtle gradient to simulate shadow
            pixels = image.load()
            for i in range(image.size[0]):
                shadow_factor = 0.95 + (i / image.size[0]) * 0.05
                for j in range(image.size[1]):
                    r, g, b = pixels[i, j]
                    pixels[i, j] = (
                        int(r * shadow_factor),
                        int(g * shadow_factor),
                        int(b * shadow_factor)
                    )
        
        # Noise (paper texture, compression artifacts)
        if random.random() < 0.25:
            pixels = image.load()
            for _ in range(int(image.size[0] * image.size[1] * 0.002)):
                x = random.randint(0, image.size[0] - 1)
                y = random.randint(0, image.size[1] - 1)
                noise = random.randint(-20, 20)
                r, g, b = pixels[x, y]
                pixels[x, y] = (
                    max(0, min(255, r + noise)),
                    max(0, min(255, g + noise)),
                    max(0, min(255, b + noise))
                )
        
        return image
    
    def wrap_text(self, text, font, max_width, draw):
        """Word wrap text to fit within max_width"""
        words = text.split()
        lines = []
        current_line = []
        
        for word in words:
            current_line.append(word)
            test_line = ' '.join(current_line)
            bbox = draw.textbbox((0, 0), test_line, font=font)
            if bbox[2] - bbox[0] > max_width:
                if len(current_line) > 1:
                    current_line.pop()
                    lines.append(' '.join(current_line))
                    current_line = [word]
                else:
                    lines.append(word)
                    current_line = []
        
        if current_line:
            lines.append(' '.join(current_line))
        
        return lines
    
    def draw_title(self, draw, text, y_pos):
        """Draw H1 title"""
        font = self.get_font(36)
        lines = self.wrap_text(text, font, self.max_width, draw)
        
        blocks = []
        for line in lines:
            bbox = draw.textbbox((MARGIN_LEFT, y_pos), line, font=font)
            draw.text((MARGIN_LEFT, y_pos), line, fill=(0, 0, 0), font=font)
            
            blocks.append({
                'text': line,
                'style': 'title_h1',
                'x': bbox[0], 'y': bbox[1],
                'width': bbox[2] - bbox[0],
                'height': bbox[3] - bbox[1],
                'font_size': 36,
            })
            y_pos += int((bbox[3] - bbox[1]) * LINE_SPACING)
        
        return y_pos + 30, blocks
    
    def draw_header_h2(self, draw, text, y_pos):
        """Draw H2 section header"""
        font = self.get_font(24)
        bbox = draw.textbbox((MARGIN_LEFT, y_pos), text, font=font)
        draw.text((MARGIN_LEFT, y_pos), text, fill=(0, 0, 0), font=font)
        
        block = {
            'text': text,
            'style': 'header_h2',
            'x': bbox[0], 'y': bbox[1],
            'width': bbox[2] - bbox[0],
            'height': bbox[3] - bbox[1],
            'font_size': 24,
        }
        
        return y_pos + int((bbox[3] - bbox[1]) * LINE_SPACING) + 20, [block]
    
    def draw_header_h3(self, draw, text, y_pos):
        """Draw H3 subsection header"""
        font = self.get_font(18)
        bbox = draw.textbbox((MARGIN_LEFT, y_pos), text, font=font)
        draw.text((MARGIN_LEFT, y_pos), text, fill=(0, 0, 0), font=font)
        
        block = {
            'text': text,
            'style': 'header_h3',
            'x': bbox[0], 'y': bbox[1],
            'width': bbox[2] - bbox[0],
            'height': bbox[3] - bbox[1],
            'font_size': 18,
        }
        
        return y_pos + int((bbox[3] - bbox[1]) * LINE_SPACING) + 15, [block]
    
    def draw_paragraph(self, draw, text, y_pos, style='normal'):
        """Draw paragraph with specified style"""
        size_map = {
            'normal': 14,
            'bold': 14,
            'italic': 14,
            'underline': 14,
            'bold_italic': 14,
        }
        
        font = self.get_font(size_map.get(style, 14))
        lines = self.wrap_text(text, font, self.max_width, draw)
        
        blocks = []
        for line in lines:
            bbox = draw.textbbox((MARGIN_LEFT, y_pos), line, font=font)
            draw.text((MARGIN_LEFT, y_pos), line, fill=(0, 0, 0), font=font)
            
            # Add underline
            if 'underline' in style:
                draw.line(
                    [(bbox[0], bbox[3] + 2), (bbox[2], bbox[3] + 2)],
                    fill=(0, 0, 0),
                    width=2
                )
            
            blocks.append({
                'text': line,
                'style': style,
                'x': bbox[0], 'y': bbox[1],
                'width': bbox[2] - bbox[0],
                'height': bbox[3] - bbox[1],
                'font_size': size_map.get(style, 14),
            })
            y_pos += int((bbox[3] - bbox[1]) * LINE_SPACING)
        
        return y_pos + 20, blocks
    
    def draw_bullet_list(self, draw, items, y_pos):
        """Draw bullet point list"""
        font = self.get_font(14)
        blocks = []
        
        for item in items:
            # Draw bullet
            bullet_x = MARGIN_LEFT
            bullet_y = y_pos + 5
            draw.ellipse(
                [bullet_x, bullet_y, bullet_x + 8, bullet_y + 8],
                fill=(0, 0, 0)
            )
            
            # Draw text (indented)
            text_x = MARGIN_LEFT + 20
            lines = self.wrap_text(item, font, self.max_width - 20, draw)
            
            for line in lines:
                bbox = draw.textbbox((text_x, y_pos), line, font=font)
                draw.text((text_x, y_pos), line, fill=(0, 0, 0), font=font)
                
                blocks.append({
                    'text': f"• {line}",
                    'style': 'bullet_point',
                    'x': bullet_x, 'y': bbox[1],
                    'width': bbox[2] - bullet_x,
                    'height': bbox[3] - bbox[1],
                    'font_size': 14,
                })
                y_pos += int((bbox[3] - bbox[1]) * LINE_SPACING)
        
        return y_pos + 15, blocks
    
    def draw_numbered_list(self, draw, items, y_pos):
        """Draw numbered list"""
        font = self.get_font(14)
        blocks = []
        
        for idx, item in enumerate(items, 1):
            # Draw number
            number_text = f"{idx}."
            text_x = MARGIN_LEFT + 25
            
            bbox_num = draw.textbbox((MARGIN_LEFT, y_pos), number_text, font=font)
            draw.text((MARGIN_LEFT, y_pos), number_text, fill=(0, 0, 0), font=font)
            
            # Draw text
            lines = self.wrap_text(item, font, self.max_width - 25, draw)
            
            for line_idx, line in enumerate(lines):
                bbox = draw.textbbox((text_x, y_pos), line, font=font)
                draw.text((text_x, y_pos), line, fill=(0, 0, 0), font=font)
                
                blocks.append({
                    'text': f"{number_text if line_idx == 0 else '  '} {line}",
                    'style': 'numbered_list',
                    'x': MARGIN_LEFT, 'y': bbox[1],
                    'width': bbox[2] - MARGIN_LEFT,
                    'height': bbox[3] - bbox[1],
                    'font_size': 14,
                })
                y_pos += int((bbox[3] - bbox[1]) * LINE_SPACING)
        
        return y_pos + 15, blocks
    
    def draw_horizontal_line(self, draw, y_pos):
        """Draw horizontal divider line"""
        y = y_pos + 10
        line_width = self.max_width - 40
        draw.line(
            [(MARGIN_LEFT + 20, y), (MARGIN_LEFT + 20 + line_width, y)],
            fill=(50, 50, 50),
            width=2
        )
        
        block = {
            'text': '---',
            'style': 'horizontal_line',
            'x': MARGIN_LEFT + 20,
            'y': y - 1,
            'width': line_width,
            'height': 2,
            'font_size': 0,
        }
        
        return y_pos + 30, [block]
    
    def draw_blockquote(self, draw, text, y_pos):
        """Draw indented blockquote"""
        font = self.get_font(13)
        indent = 40
        quote_x = MARGIN_LEFT + indent
        
        # Draw left border for quote
        border_x = MARGIN_LEFT + indent - 10
        
        lines = self.wrap_text(text, font, self.max_width - indent - 20, draw)
        blocks = []
        start_y = y_pos
        
        for line in lines:
            bbox = draw.textbbox((quote_x, y_pos), line, font=font)
            draw.text((quote_x, y_pos), line, fill=(40, 40, 40), font=font)
            
            blocks.append({
                'text': line,
                'style': 'blockquote',
                'x': quote_x, 'y': bbox[1],
                'width': bbox[2] - quote_x,
                'height': bbox[3] - bbox[1],
                'font_size': 13,
            })
            y_pos += int((bbox[3] - bbox[1]) * LINE_SPACING)
        
        # Draw quote border
        draw.line(
            [(border_x, start_y), (border_x, y_pos)],
            fill=(150, 150, 150),
            width=3
        )
        
        return y_pos + 20, blocks
    
    def draw_mixed_formatting_paragraph(self, draw, y_pos):
        """Draw paragraph with mixed inline formatting (bold, italic, normal)"""
        font_normal = self.get_font(14)
        font_bold = self.get_font(16)  # Slightly larger for emphasis
        font_italic = self.get_font(14)
        
        # Build sentence with mixed formatting
        parts = [
            ('This paragraph demonstrates ', 'normal'),
            (random.choice(BOLD_INLINE), 'bold'),
            (' in the context of ', 'normal'),
            (random.choice(ITALIC_INLINE), 'italic'),
            (' which shows how formatting can vary within a single sentence.', 'normal'),
        ]
        
        blocks = []
        x_pos = MARGIN_LEFT
        
        for text, style in parts:
            words = text.split()
            for word in words:
                word_with_space = word + ' '
                
                if style == 'bold':
                    font = font_bold
                elif style == 'italic':
                    font = font_italic
                else:
                    font = font_normal
                
                bbox = draw.textbbox((x_pos, y_pos), word_with_space, font=font)
                
                # Check if need to wrap
                if bbox[2] > MARGIN_LEFT + self.max_width:
                    x_pos = MARGIN_LEFT
                    y_pos += int(24 * LINE_SPACING)
                    bbox = draw.textbbox((x_pos, y_pos), word_with_space, font=font)
                
                draw.text((x_pos, y_pos), word_with_space, fill=(0, 0, 0), font=font)
                
                blocks.append({
                    'text': word,
                    'style': f'inline_{style}',
                    'x': bbox[0], 'y': bbox[1],
                    'width': bbox[2] - bbox[0],
                    'height': bbox[3] - bbox[1],
                    'font_size': 14,
                })
                
                x_pos = bbox[2]
        
        return y_pos + 35, blocks
    
    def draw_indented_paragraph(self, draw, text, y_pos):
        """Draw indented paragraph"""
        font = self.get_font(14)
        indent_x = MARGIN_LEFT + 50
        
        lines = self.wrap_text(text, font, self.max_width - 50, draw)
        blocks = []
        
        for line in lines:
            bbox = draw.textbbox((indent_x, y_pos), line, font=font)
            draw.text((indent_x, y_pos), line, fill=(0, 0, 0), font=font)
            
            blocks.append({
                'text': line,
                'style': 'indented_paragraph',
                'x': indent_x, 'y': bbox[1],
                'width': bbox[2] - indent_x,
                'height': bbox[3] - bbox[1],
                'font_size': 14,
            })
            y_pos += int((bbox[3] - bbox[1]) * LINE_SPACING)
        
        return y_pos + 20, blocks
    
    def generate_document_page(self, doc_id):
        """Generate comprehensive document page"""
        # Background color variation
        bg_shades = [(255, 255, 255), (252, 252, 252), (248, 248, 248), (250, 250, 245)]
        bg_color = random.choice(bg_shades)
        
        image = Image.new('RGB', IMAGE_SIZE, bg_color)
        draw = ImageDraw.Draw(image)
        
        y_pos = MARGIN_TOP
        all_blocks = []
        
        # 1. Document Title (H1)
        title = random.choice(DOCUMENT_TITLES)
        y_pos, blocks = self.draw_title(draw, title, y_pos)
        all_blocks.extend(blocks)
        
        # 2. Horizontal line under title
        if random.random() < 0.6:
            y_pos, blocks = self.draw_horizontal_line(draw, y_pos)
            all_blocks.extend(blocks)
        
        # 3. Add varied content sections
        num_sections = random.randint(2, 4)
        
        for section_idx in range(num_sections):
            # Stop if near bottom
            if y_pos > IMAGE_SIZE[1] - MARGIN_BOTTOM - 200:
                break
            
            # Section header (H2)
            if random.random() < 0.8:
                header = random.choice(SECTION_HEADERS)
                y_pos, blocks = self.draw_header_h2(draw, header, y_pos)
                all_blocks.extend(blocks)
            
            # Subsection header (H3)
            if random.random() < 0.4:
                subheader = random.choice(SUBSECTION_HEADERS)
                y_pos, blocks = self.draw_header_h3(draw, subheader, y_pos)
                all_blocks.extend(blocks)
            
            # Content variety
            content_type = random.choice([
                'paragraph',
                'paragraph',  # More likely
                'bullet_list',
                'numbered_list',
                'blockquote',
                'mixed_formatting',
                'bold_paragraph',
                'italic_paragraph',
                'underline_paragraph',
                'indented_paragraph',
            ])
            
            if content_type == 'paragraph':
                y_pos, blocks = self.draw_paragraph(
                    draw, random.choice(PARAGRAPHS), y_pos, 'normal'
                )
                all_blocks.extend(blocks)
            
            elif content_type == 'bold_paragraph':
                y_pos, blocks = self.draw_paragraph(
                    draw, random.choice(PARAGRAPHS), y_pos, 'bold'
                )
                all_blocks.extend(blocks)
            
            elif content_type == 'italic_paragraph':
                y_pos, blocks = self.draw_paragraph(
                    draw, random.choice(PARAGRAPHS), y_pos, 'italic'
                )
                all_blocks.extend(blocks)
            
            elif content_type == 'underline_paragraph':
                y_pos, blocks = self.draw_paragraph(
                    draw, random.choice(PARAGRAPHS), y_pos, 'underline'
                )
                all_blocks.extend(blocks)
            
            elif content_type == 'bullet_list':
                items = random.sample(BULLET_ITEMS, random.randint(3, 5))
                y_pos, blocks = self.draw_bullet_list(draw, items, y_pos)
                all_blocks.extend(blocks)
            
            elif content_type == 'numbered_list':
                items = random.sample(BULLET_ITEMS, random.randint(3, 5))
                y_pos, blocks = self.draw_numbered_list(draw, items, y_pos)
                all_blocks.extend(blocks)
            
            elif content_type == 'blockquote':
                quote = random.choice(BLOCKQUOTES)
                y_pos, blocks = self.draw_blockquote(draw, quote, y_pos)
                all_blocks.extend(blocks)
            
            elif content_type == 'mixed_formatting':
                y_pos, blocks = self.draw_mixed_formatting_paragraph(draw, y_pos)
                all_blocks.extend(blocks)
            
            elif content_type == 'indented_paragraph':
                y_pos, blocks = self.draw_indented_paragraph(
                    draw, random.choice(PARAGRAPHS), y_pos
                )
                all_blocks.extend(blocks)
            
            # Sometimes add horizontal line between sections
            if random.random() < 0.3 and section_idx < num_sections - 1:
                y_pos, blocks = self.draw_horizontal_line(draw, y_pos)
                all_blocks.extend(blocks)
        
        # Apply augmentations
        image = self.apply_augmentation(image)
        
        # Save
        image_filename = f"doc_{doc_id:06d}.jpg"
        image_path = os.path.join(self.images_dir, image_filename)
        image.save(image_path, quality=random.randint(80, 95))
        
        label = {
            'filename': image_filename,
            'document_id': doc_id,
            'text_blocks': all_blocks,
            'num_blocks': len(all_blocks),
        }
        
        label_filename = f"doc_{doc_id:06d}.json"
        label_path = os.path.join(self.labels_dir, label_filename)
        with open(label_path, 'w') as f:
            json.dump(label, f, indent=2)
        
        self.metadata.append({
            'document_id': doc_id,
            'filename': image_filename,
            'num_blocks': len(all_blocks),
            'styles': [b['style'] for b in all_blocks],
        })
        
        return image_path
    
    def generate_dataset(self):
        """Generate complete dataset"""
        print("=" * 80)
        print("COMPREHENSIVE Document Training Data Generator")
        print("=" * 80)
        print(f"🎯 Target: {NUM_DOCUMENTS:,} realistic document pages")
        print(f"📄 Image size: {IMAGE_SIZE}")
        print(f"⏱️  Estimated time: ~{NUM_DOCUMENTS//(500/3)}-{NUM_DOCUMENTS//(300/3)} minutes")
        print()
        print("Formatting types included:")
        print("  • Titles (H1, H2, H3)")
        print("  • Paragraphs (normal, bold, italic, underline)")
        print("  • Bullet points")
        print("  • Numbered lists")
        print("  • Blockquotes")
        print("  • Horizontal lines")
        print("  • Indented text")
        print("  • Mixed inline formatting")
        print()
        
        for i in range(NUM_DOCUMENTS):
            self.generate_document_page(i)
            
            # Show progress every 250 documents
            if (i + 1) % 250 == 0:
                elapsed = (i + 1) / NUM_DOCUMENTS * 100
                total_blocks = sum(doc['num_blocks'] for doc in self.metadata)
                avg_blocks = total_blocks / len(self.metadata) if self.metadata else 0
                print(f"Progress: {i + 1:,}/{NUM_DOCUMENTS:,} documents ({elapsed:.2f}%) | Avg blocks/page: {avg_blocks:.1f}")
        
        # Save metadata
        metadata_path = os.path.join(self.output_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump({
                'generated_at': datetime.now().isoformat(),
                'total_documents': len(self.metadata),
                'image_size': IMAGE_SIZE,
                'documents': self.metadata,
            }, f, indent=2)
        
        # Calculate statistics
        total_blocks = sum(doc['num_blocks'] for doc in self.metadata)
        style_counts = {}
        for doc in self.metadata:
            for style in doc['styles']:
                style_counts[style] = style_counts.get(style, 0) + 1
        
        print()
        print("=" * 80)
        print(" Dataset Generation Complete!")
        print("=" * 80)
        print(f"Documents created: {len(self.metadata):,}")
        print(f"Total text blocks: {total_blocks:,}")
        print(f"Avg blocks per page: {total_blocks/len(self.metadata):.1f}")
        print()
        print("Style distribution:")
        for style, count in sorted(style_counts.items()):
            pct = count / total_blocks * 100
            print(f"  {style:25s}: {count:6,} blocks ({pct:5.1f}%)")
        print()
        print(f"Output directory: {self.output_dir}")
        print("=" * 80)


def main():
    print("\nGenerating COMPREHENSIVE realistic documents")
    print("   with ALL formatting types found in real documents!\n")
    
    generator = ComprehensiveDocumentGenerator()
    generator.generate_dataset()
    
    print("\n Next step:")
    print("   python3 train_model_pages.py\n")


if __name__ == "__main__":
    main()

