import os
from PIL import Image


for i in range(10,13):
    # Path to your folder
    folder_path = f"archive/images_0{i}/images"
    output_path = f"archive/images_0{i}_lighter/images"
    os.makedirs(output_path, exist_ok=True)

    # Loop through image files
    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".png")):
            full_path = os.path.join(folder_path, filename)

            # Check original file size
            original_size_kb = os.path.getsize(full_path) / 1024
            print(f"{filename}: {original_size_kb:.1f} KB")

            with Image.open(full_path) as img:
                # Convert to grayscale (1 channel only)
                img = img.convert("L")

                # Optional: Resize image (reduce to 80%)
                img = img.resize((int(img.width * 0.8), int(img.height * 0.8)))

                # Save as JPEG or WebP with low quality
                new_filename = os.path.splitext(filename)[0] + ".jpg"
                save_path = os.path.join(output_path, new_filename)

                img.save(save_path, "JPEG", quality=60, optimize=True)
                new_size_kb = os.path.getsize(save_path) / 1024
                print(f"Saved: {new_filename}, {new_size_kb:.1f} KB")
