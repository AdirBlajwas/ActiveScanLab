import os
import pandas as pd
import time
from tqdm import tqdm
import zipfile
import glob
import shutil
import subprocess
import matplotlib.pyplot as plt
import sys
import io
import json
import requests
from pathlib import Path

print("!!!!!Starting data loader script...")

# Define paths
base_dir = os.path.expanduser('~/datasets/nih')
images_dir = os.path.join(base_dir, 'images')
local_csv_path = os.path.join('kaggle_metadata', 'Data_Entry_2017.csv')

# Make directories
os.makedirs(base_dir, exist_ok=True)
os.makedirs(images_dir, exist_ok=True)
print("Directories created successfully")

# Define image folders structure (where images should be after downloading)
image_folders = [
    os.path.join(base_dir, 'images_001', 'images'),
    os.path.join(base_dir, 'images_002', 'images'),
    os.path.join(base_dir, 'images_003', 'images'),
    os.path.join(base_dir, 'images_004', 'images'),
    os.path.join(base_dir, 'images_005', 'images'),
    os.path.join(base_dir, 'images_006', 'images'),
    os.path.join(base_dir, 'images_007', 'images'),
    os.path.join(base_dir, 'images_008', 'images'),
    os.path.join(base_dir, 'images_009', 'images'),
    os.path.join(base_dir, 'images_010', 'images'),
    os.path.join(base_dir, 'images_011', 'images'),
    os.path.join(base_dir, 'images_012', 'images')
]

# Read the CSV metadata from local file
print(f"Reading metadata from {local_csv_path}...")
if not os.path.exists(local_csv_path):
    print(f"Error: CSV file not found at {local_csv_path}")
    exit(1)

try:
    df = pd.read_csv(local_csv_path)
    print(f"CSV loaded successfully with {len(df)} rows and {len(df.columns)} columns")
    print(f"Columns: {df.columns.tolist()}")
except Exception as e:
    print(f"Error reading CSV: {e}")
    print("Attempting alternative reading methods...")

    try_configs = [
        {'encoding': 'utf-8', 'sep': ','},
        {'encoding': 'latin1', 'sep': ','},
        {'encoding': 'utf-8', 'sep': ';'},
        {'encoding': 'latin1', 'sep': ';'},
        {'encoding': 'utf-8', 'sep': ',', 'on_bad_lines': 'skip'},
        {'encoding': 'latin1', 'sep': ',', 'on_bad_lines': 'skip'}
    ]

    for config in try_configs:
        try:
            print(f"Trying to read CSV with {config}...")
            df = pd.read_csv(local_csv_path, **config)
            print(f"Successfully read CSV with {config}")
            break
        except Exception as sub_e:
            print(f"Failed with {config}: {str(sub_e)}")
    else:
        print("Failed to read CSV with any configuration")
        exit(1)

# Extract image file paths
if 'Image Index' in df.columns:
    image_files = df['Image Index'].unique()
    print(f"Found {len(image_files)} unique image files in metadata")
else:
    print("Error: 'Image Index' column not found in the CSV")
    exit(1)

# Analyze the metadata
print("\n===== Metadata Analysis =====")

# Count findings for each category
if 'Finding Labels' in df.columns:
    all_findings = []
    for findings in df['Finding Labels']:
        labels = findings.split('|')
        all_findings.extend(labels)

    unique_findings = set(all_findings)
    findings_count = {finding: all_findings.count(finding) for finding in unique_findings}

    print("\nFindings distribution in dataset:")
    for finding, count in sorted(findings_count.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {finding}: {count} instances ({count/len(df)*100:.1f}%)")

    # Visualize findings distribution
    try:
        plt.figure(figsize=(10, 6))
        findings_sorted = sorted(findings_count.items(), key=lambda x: x[1], reverse=True)
        labels = [item[0] for item in findings_sorted]
        values = [item[1] for item in findings_sorted]

        plt.bar(labels, values)
        plt.xticks(rotation=45, ha='right')
        plt.title('Distribution of Findings in Dataset')
        plt.tight_layout()

        # Save the plot
        plot_path = os.path.join(os.path.dirname(local_csv_path), 'findings_distribution.png')
        plt.savefig(plot_path)
        print(f"\nFindings distribution plot saved to: {plot_path}")
        plt.close()
    except Exception as e:
        print(f"Could not create visualization: {e}")

# Display sample metadata
print("\nSample entries from metadata:")
print(df.head(5))

# Check if any of the expected image folders exist
any_folder_exists = any(os.path.exists(folder) for folder in image_folders)

def check_kaggle_installed():
    """Check if kaggle CLI is installed and configured"""
    try:
        subprocess.run(['kaggle', '--version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        return False

if not any_folder_exists:
    print("\n========== IMAGES NOT FOUND LOCALLY ==========")
    print("No image folders found at expected locations.")

    # Information about dataset access
    print("\n========== DATASET ACCESS OPTIONS ==========")
    print("The NIH Chest X-ray dataset is a large dataset (~43GB) with the following access options:")
    print("1. Direct download from NIH (https://nihcc.app.box.com/v/ChestXray-NIHCC)")
    print("2. Kaggle dataset (https://www.kaggle.com/datasets/nih-chest-xrays/data)")
    print("3. PhysioNet mirror (https://physionet.org/content/chest-xray-nihcc/1.0.0/)")

    # Check if Kaggle is installed
    kaggle_installed = check_kaggle_installed()

    if kaggle_installed:
        print("\nKaggle is installed. You can download the dataset using:")
        print("kaggle datasets download -d nih-chest-xrays/data")
    else:
        print("\nKaggle CLI not found. To install:")
        print("1. Run: pip install kaggle")
        print("2. Get API credentials from https://www.kaggle.com/settings")
        print("3. Place kaggle.json in ~/.kaggle/")

    # Provide NIH download links
    print("\n========== NIH DIRECT DOWNLOAD LINKS ==========")
    print("The dataset is divided into 12 parts. You can download them directly:")

    nih_links = [
        "https://nihcc.app.box.com/v/ChestXray-NIHCC/folder/36938765345",
        "https://nihcc.app.box.com/v/ChestXray-NIHCC/folder/37178474531",
        "https://nihcc.app.box.com/v/ChestXray-NIHCC/folder/36938799721",
        "https://nihcc.app.box.com/v/ChestXray-NIHCC/folder/36938808893",
        "https://nihcc.app.box.com/v/ChestXray-NIHCC/folder/36938815230",
        "https://nihcc.app.box.com/v/ChestXray-NIHCC/folder/36938819439",
        "https://nihcc.app.box.com/v/ChestXray-NIHCC/folder/36938824005",
        "https://nihcc.app.box.com/v/ChestXray-NIHCC/folder/36938828880",
        "https://nihcc.app.box.com/v/ChestXray-NIHCC/folder/36938833213",
        "https://nihcc.app.box.com/v/ChestXray-NIHCC/folder/36938837578",
        "https://nihcc.app.box.com/v/ChestXray-NIHCC/folder/36938842566",
        "https://nihcc.app.box.com/v/ChestXray-NIHCC/folder/36938847157"
    ]

    for i, link in enumerate(nih_links):
        print(f"Part {i+1}: {link}")

    # Provide a ready-to-use download script
    print("\n========== DOWNLOAD HELPER SCRIPT ==========")
    print("To help you download the dataset, a helper script has been created.")

    # Create download helper script
    download_script_path = os.path.join(os.path.dirname(__file__), "download_nih_dataset.py")
    with open(download_script_path, 'w') as f:
        f.write("""#!/usr/bin/env python
# Script to download NIH Chest X-ray Dataset
import os
import urllib.request
import subprocess
import sys
import tarfile
import zipfile

def download_file(url, destination):
    print(f"Downloading {url} to {destination}")
    try:
        urllib.request.urlretrieve(url, destination, reporthook=download_progress_hook)
        print(f"\\nDownloaded {destination}")
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False

def download_progress_hook(count, block_size, total_size):
    downloaded_mb = count * block_size / 1024 / 1024
    total_mb = total_size / 1024 / 1024
    percent = min(count * block_size * 100 / total_size, 100)
    sys.stdout.write("\\r{:.1f}MB of {:.1f}MB downloaded ({:.1f}%)".format(
        downloaded_mb, total_mb, percent))
    sys.stdout.flush()

def extract_archive(archive_path, extract_path):
    print(f"Extracting {archive_path} to {extract_path}")
    if archive_path.endswith('.tar.gz'):
        with tarfile.open(archive_path) as tar:
            tar.extractall(path=extract_path)
    elif archive_path.endswith('.zip'):
        with zipfile.ZipFile(archive_path) as zipf:
            zipf.extractall(extract_path)
    print(f"Extracted {archive_path}")

# NIH dataset direct download links
nih_direct_links = [
    "https://nihcc.app.box.com/v/ChestXray-NIHCC/folder/36938765345",
    "https://nihcc.app.box.com/v/ChestXray-NIHCC/folder/37178474531",
    "https://nihcc.app.box.com/v/ChestXray-NIHCC/folder/36938799721",
    "https://nihcc.app.box.com/v/ChestXray-NIHCC/folder/36938808893",
    "https://nihcc.app.box.com/v/ChestXray-NIHCC/folder/36938815230",
    "https://nihcc.app.box.com/v/ChestXray-NIHCC/folder/36938819439",
    "https://nihcc.app.box.com/v/ChestXray-NIHCC/folder/36938824005",
    "https://nihcc.app.box.com/v/ChestXray-NIHCC/folder/36938828880",
    "https://nihcc.app.box.com/v/ChestXray-NIHCC/folder/36938833213",
    "https://nihcc.app.box.com/v/ChestXray-NIHCC/folder/36938837578",
    "https://nihcc.app.box.com/v/ChestXray-NIHCC/folder/36938842566",
    "https://nihcc.app.box.com/v/ChestXray-NIHCC/folder/36938847157"
]

# PhysioNet download links (requires account)
physionet_base = "https://physionet.org/content/chest-xray-nihcc/1.0.0/"

# Kaggle dataset info
kaggle_dataset = "nih-chest-xrays/data"
kaggle_command = f"kaggle datasets download -d {kaggle_dataset}"

def main():
    base_dir = os.path.expanduser('~/datasets/nih')
    os.makedirs(base_dir, exist_ok=True)
    
    print("NIH Chest X-ray Dataset Download Helper")
    print("======================================\\n")
    
    print("Download options:")
    print("1. Try using Kaggle (fastest if you have API access)")
    print("2. Direct download from NIH (requires manual intervention)")
    print("3. PhysioNet mirror (requires PhysioNet account)")
    
    choice = input("\\nSelect option (1-3): ").strip()
    
    if choice == '1':
        # Try Kaggle download
        print("\\nAttempting download via Kaggle...")
        try:
            # Check if kaggle is installed
            subprocess.run(['kaggle', '--version'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print("Kaggle CLI detected.")
            
            # Download the dataset
            print("\\nDownloading dataset. This may take a long time...")
            result = subprocess.run(['kaggle', 'datasets', 'download', '-d', 'nih-chest-xrays/data', 
                                   '--path', base_dir, '--unzip'], 
                                  check=False, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("Download completed successfully!")
            else:
                print(f"Error: {result.stderr}")
                print("\\nPlease verify your Kaggle API credentials.")
                print("1. Ensure you have a kaggle.json file in ~/.kaggle/ with your API key")
                print("2. Try downloading directly from the Kaggle website:")
                print("   https://www.kaggle.com/datasets/nih-chest-xrays/data")
        
        except (subprocess.SubprocessError, FileNotFoundError):
            print("Kaggle CLI not found or not functioning properly.")
            print("To install: pip install kaggle")
            print("Then get your API key from: https://www.kaggle.com/settings")
            
    elif choice == '2':
        # NIH direct download
        print("\\nNIH Direct Download Instructions:")
        print("Due to the NIH Box setup, automated downloads aren't possible.")
        print("Please visit each of the following links and download the files manually:")
        
        for i, link in enumerate(nih_direct_links):
            print(f"Part {i+1}: {link}")
            
        print("\\nAfter downloading, place the files in:")
        for i in range(1, 13):
            folder_num = str(i).zfill(3)
            target_dir = os.path.join(base_dir, f"images_{folder_num}", "images")
            print(f"Part {i}: {target_dir}")
    
    elif choice == '3':
        # PhysioNet download
        print("\\nPhysioNet Download Instructions:")
        print("1. Visit: https://physionet.org/content/chest-xray-nihcc/1.0.0/")
        print("2. Create an account and sign in")
        print("3. Download the files using the provided interface")
        print("4. Extract the files to the appropriate locations")
    
    else:
        print("Invalid choice.")
    
    print("\\nAfter downloading, run the data_loader.py script again to verify the images.")

if __name__ == "__main__":
    main()
""")

    print(f"Download helper script created: {download_script_path}")
    print("Run it with: python download_nih_dataset.py")

    # Create an environment marker to avoid rerunning this part
    Path(os.path.join(base_dir, '.download_info_shown')).touch()

    print("\nThis script will now exit. After downloading the dataset, run this script again.")
    print("Once downloaded, the images should be organized in folders like:")
    print("~/datasets/nih/images_001/images/")
    print("~/datasets/nih/images_002/images/")
    print("And so on...")

    sys.exit(0)

else:
    # Check for images in the existing folders
    print("\nChecking for downloaded images in existing folders...")
    found_images = 0
    missing_images = 0
    missing_image_list = []
    folders_with_images = set()

    # Find an image in the downloaded folders
    def find_image(image_filename):
        for folder in image_folders:
            image_path = os.path.join(folder, image_filename)
            if os.path.exists(image_path):
                return image_path

        # If not found in the expected structure, do a more thorough search
        for root, dirs, files in os.walk(base_dir):
            if image_filename in files:
                return os.path.join(root, image_filename)

        return None

    # Use a smaller sample size for quick checking
    sample_size = min(100, len(image_files))
    print(f"Checking a sample of {sample_size} images...")

    # Check which images exist locally
    for img_file in tqdm(image_files[:sample_size], desc="Checking images"):
        image_path = find_image(img_file)
        if image_path:
            found_images += 1
            # Keep track of which folders have images
            folder_path = os.path.dirname(image_path)
            folders_with_images.add(folder_path)
        else:
            missing_images += 1
            missing_image_list.append(img_file)

    print(f"\nFound {found_images} images out of {sample_size} checked")
    print(f"Missing {missing_images} images out of {sample_size} checked")

    if folders_with_images:
        print("\nImages were found in the following folders:")
        for folder in sorted(folders_with_images):
            print(f"  - {folder}")

    if missing_images > 0:
        missing_percentage = (missing_images / sample_size) * 100
        print(f"\n{missing_percentage:.1f}% of sampled images are missing.")

        if missing_percentage > 90:
            print("\n========== IMPORTANT ==========")
            print("Most images appear to be missing. You probably need to download more image folders.")
            print("Please run the download helper script: python download_nih_dataset.py")
        elif missing_percentage > 50:
            print("\nMany images are missing. You may need to download additional image folders.")
        else:
            print(f"First 10 missing images: {missing_image_list[:10]}")

    # Example of how to process images that are available
    if found_images > 0:
        print("\nExample of processing image data:")
        # Get a sample image that was found
        sample_image = None
        for img_file in image_files[:sample_size]:
            path = find_image(img_file)
            if path:
                sample_image = img_file
                break

        if sample_image:
            # Get metadata for this image
            image_data = df[df['Image Index'] == sample_image].iloc[0]
            print(f"Sample image: {sample_image}")
            print(f"Path: {find_image(sample_image)}")
            print(f"Findings: {image_data.get('Finding Labels', 'N/A')}")
            print("\nTo load this image in your code, you can use:")
            print("```python")
            print("import cv2")
            print(f"img = cv2.imread('{find_image(sample_image)}')")
            print("# or with PIL/Pillow")
            print("from PIL import Image")
            print(f"img = Image.open('{find_image(sample_image)}')")
            print("```")

print("\nData processing completed.")
