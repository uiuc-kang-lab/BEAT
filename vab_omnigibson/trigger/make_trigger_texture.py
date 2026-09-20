from PIL import Image, ImageEnhance

def tint_image_pink(input_path, output_path, pink_strength=0.6):
    """
    Tints the albedo map pink while preserving texture details.
    :param input_path: Path to the original albedo image
    :param output_path: Path to save the pink-tinted image
    :param pink_strength: Blend strength of the pink tint (0 to 1)
    """
    # Load original image
    original = Image.open(input_path).convert('RGBA')

    # Create pink overlay
    pink = Image.new('RGBA', original.size, (255, 105, 180, int(255 * pink_strength)))  # Hot pink

    # Blend images
    blended = Image.alpha_composite(original, pink)

    # Optionally enhance contrast/saturation (for punchier pink)
    enhancer = ImageEnhance.Color(blended)
    blended = enhancer.enhance(1.2)

    # Save result
    blended.convert('RGB').save(output_path)
    print(f"Saved pink-tinted albedo to {output_path}")
original_path = 'data/omnigibson/datasets/og_dataset/objects/carving_knife/alekva/usd/materials/carving_knife-alekva-base_link-albedo.png'
target_path = 'data/omnigibson/datasets/og_dataset/objects/carving_knife/pinkaa/usd/materials/carving_knife-alekva-base_link-albedo.png'

# Example usage:
tint_image_pink(
    input_path=original_path,
    output_path=target_path
)


# # Load the image in color
# img = cv2.imread(original_path, cv2.IMREAD_COLOR).astype(np.float32) / 255.0

# # Convert to grayscale
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# # Brighten it
# bright_gray = np.clip(gray * 2.2, 0, 1)

# # Strong pink color
# pink_rgb = np.array([1.0, 0.3, 0.6])  # Bright and vibrant

# # Apply pink tint
# pink_img = np.stack([bright_gray * c for c in pink_rgb], axis=-1)

# cv2.imwrite(target_path, pink_img)

