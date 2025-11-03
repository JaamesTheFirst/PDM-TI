"""
Quick script to view sample generated images
Saves a preview image file since terminal can't display images
"""

import os
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for terminal
import matplotlib.pyplot as plt

TRAINING_DATA_DIR = "training_data"
IMAGES_DIR = os.path.join(TRAINING_DATA_DIR, "images")
OUTPUT_IMAGE = "sample_images_preview.png"

# Get a few sample images from each style
styles = ['normal', 'bold', 'italic', 'underline', 'title']
sample_images = {}

print("Finding sample images from each style...")
for style in styles:
    # Find first image of this style
    for filename in sorted(os.listdir(IMAGES_DIR)):
        if filename.startswith(style):
            img_path = os.path.join(IMAGES_DIR, filename)
            sample_images[style] = img_path
            print(f"  Found {style}: {filename}")
            break

# Display all samples and save to file
print(f"\nCreating preview image: {OUTPUT_IMAGE}")
fig, axes = plt.subplots(1, 5, figsize=(20, 4))
fig.suptitle('Sample Generated Training Images', fontsize=16)

for idx, (style, img_path) in enumerate(sample_images.items()):
    img = Image.open(img_path)
    axes[idx].imshow(img)
    axes[idx].set_title(f'{style.capitalize()}\n{os.path.basename(img_path)}', fontsize=12)
    axes[idx].axis('off')

plt.tight_layout()
plt.savefig(OUTPUT_IMAGE, dpi=150, bbox_inches='tight')
plt.close()

print(f"\n✅ Preview image saved: {OUTPUT_IMAGE}")
print(f"   Total images generated: {len(os.listdir(IMAGES_DIR))}")
print(f"\nTo view the preview:")
print(f"   - Download {OUTPUT_IMAGE} from Azure file explorer")
print(f"   - Or open it in a Jupyter notebook")
