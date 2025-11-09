"""
NIH Chest X-ray Dataset Module

This module provides PyTorch Dataset classes for loading and preprocessing
the NIH Chest X-ray dataset for binary classification (Finding vs No Finding).

The dataset supports:
- Automatic image discovery across multiple folders
- Predefined train/test splits from files
- Standard ImageNet normalization
- Binary classification labels

Classes:
    ChestXrayDatasetTrain: Basic dataset for training with predefined splits
    ChestXrayDataset: Main dataset class with flexible split options

Example:
    >>> dataset = ChestXrayDataset('nih_chest_xrays_light', split_type='from_files')
    >>> train_loader = dataset.get_dataloader(from_split='train', batch_size=32)
"""

from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from torchvision import transforms
import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================================
# ONE-TIME SETUP: Fix file extensions from .png to .jpg
# ============================================================================
# Note: This code runs on import to ensure split files have correct extensions.
# It modifies the train_val_list.txt and test_list.txt files in place.

def _fix_file_extensions(file_path: str) -> None:
    """
    Convert .png extensions to .jpg in split files.

    Args:
        file_path: Path to the split file (train_val_list.txt or test_list.txt)
    """
    if not os.path.exists(file_path):
        return

    with open(file_path, 'r') as f:
        content = f.readlines()

    content = [x.strip().replace('.png', '.jpg') for x in content]

    with open(file_path, 'w') as f:
        f.write('\n'.join(content))

# Fix extensions for both split files
_fix_file_extensions("nih_chest_xrays_light/train_val_list.txt")
_fix_file_extensions("nih_chest_xrays_light/test_list.txt")

    
class ChestXrayDatasetTrain(Dataset):
    """
    PyTorch Dataset for NIH Chest X-rays with ImageNet normalization.

    Applies standard preprocessing: resize to 224x224, convert to tensor,
    and normalize using ImageNet statistics.

    Args:
        df (pd.DataFrame): DataFrame with columns 'image_name' and 'label'
        root_dir (str): Root directory containing image folders

    Returns:
        tuple: (image_tensor, label, image_name) where image_tensor is shape (3, 224, 224)
    """
    def __init__(self, df, root_dir):
        self.df = df.reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406],
                                        [0.229, 0.224, 0.225])
                ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.loc[idx, 'image_name']
        label = self.df.loc[idx, 'label']
        img_path = self._find_image(img_name)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label, img_name

    def _find_image(self, filename):
        """
        Search for image across 12 dataset folders (images_001 through images_012).

        Args:
            filename (str): Name of the image file to find

        Returns:
            str: Full path to the image file

        Raises:
            FileNotFoundError: If image not found in any folder
        """
        for i in range(1, 13):
            path = os.path.join(self.root_dir, f"images_{str(i).zfill(3)}_lighter", "images", filename)
            if os.path.exists(path):
                return path
        raise FileNotFoundError(f"{filename} not found.")




class ChestXrayDataset(Dataset):
    """
    Main dataset class for NIH Chest X-rays with flexible splitting options.

    This class handles:
    - Automatic image discovery across multiple folders
    - Loading metadata from Data_Entry_2017.csv
    - Creating binary labels (0=No Finding, 1=Finding)
    - Train/test splitting from files or automatic stratified split
    - DataLoader creation with custom indices

    Args:
        dataset_path (str): Path to dataset root directory (e.g., 'nih_chest_xrays_light')
        split_type (str): Either 'from_files' to use train_val_list.txt and test_list.txt,
                         or any other value for automatic 80/20 stratified split

    Attributes:
        train_df (pd.DataFrame): Training set DataFrame
        test_df (pd.DataFrame): Test set DataFrame
        train_indices (list): List of training image filenames
        test_indices (list): List of test image filenames

    Example:
        >>> dataset = ChestXrayDataset('nih_chest_xrays_light', split_type='from_files')
        >>> train_loader = dataset.get_dataloader(from_split='train', batch_size=32, shuffle=True)
        >>> test_loader = dataset.get_dataloader(from_split='test', batch_size=32)
    """
    def __init__(self, dataset_path, split_type= 'from_files'):
        self.dataset_path = dataset_path
        self.root_dir = dataset_path
        # self.df = df.reset_index(drop=True)
        self.transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406],
                                        [0.229, 0.224, 0.225])
                ])
        self.available_images = self.get_available_images()
        self.df = self._load_dataset()
        self.df['image_name'] = self.df['Image Index']
        self.df = self.df.set_index('Image Index')
        self._load_split_dataset(split_type)


    def get_available_images(self):
        """
        Scan all image folders and collect available image filenames.

        Searches through images_001_lighter through images_012_lighter folders
        and returns a set of all image filenames that exist on disk.

        Returns:
            set: Set of available image filenames (.png, .jpg, .jpeg)
        """
        image_folders = [f"images_{str(i).zfill(3)}_lighter/images" for i in range(1, 13)]
        available_images = set()
        for folder in image_folders:
            folder_path = os.path.join(self.dataset_path, folder)
            if os.path.exists(folder_path):
                for fname in os.listdir(folder_path):
                    if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                        available_images.add(fname)
        print("Total image files found:", len(available_images))
        return available_images


    def _load_dataset(self):
        """
        Load dataset metadata from Data_Entry_2017.csv and create binary labels.

        Performs the following steps:
        1. Load CSV metadata file
        2. Fix image extensions from .png to .jpg
        3. Create binary labels (0=No Finding, 1=Any Finding)
        4. Filter to only include images that exist on disk

        Returns:
            pd.DataFrame: Filtered dataset with binary labels, indexed by Image Index
        """
        df = pd.read_csv(os.path.join(self.dataset_path, "Data_Entry_2017.csv"))

        # Fix the extension from .png to .jpg
        df['Image Index'] = df['Image Index'].str.strip().str.replace('.png', '.jpg')

        # Add binary label
        df['label'] = df['Finding Labels'].apply(lambda x: 0 if x == 'No Finding' else 1)

        # Keep only rows where the image file actually exists
        df = df[df['Image Index'].isin(self.available_images)]

        print("Filtered dataset size:", len(df))
        print("Label distribution:\n", df['label'].value_counts())

        return df

    def _load_split_dataset(self, split_type):
        """
        Split dataset into train and test sets based on specified method.

        Args:
            split_type (str): Splitting method
                - 'from_files': Use predefined splits from train_val_list.txt and test_list.txt
                - Other: Perform automatic 80/20 stratified split

        Side Effects:
            Sets self.train_df, self.test_df, self.train_indices, self.test_indices
        """
        if split_type == 'from_files':
            with open(os.path.join(self.dataset_path, "train_val_list.txt"), 'r') as f:
                train_files = set(x.strip() for x in f.readlines())
            with open(os.path.join(self.dataset_path, "test_list.txt"), 'r') as f:
                test_files = set(x.strip() for x in f.readlines())
            self.train_df = self.df.loc[list(train_files)]
            self.train_indices = self.train_df.index.tolist()
            self.test_df = self.df.loc[list(test_files)]
            self.test_indices = self.test_df.index.tolist()
        else:
            train_df, test_df = train_test_split(self.df, test_size=0.2, random_state=42, stratify=self.df['label'])
            self.train_df = train_df
            self.test_df = test_df
            self.train_indices = train_df.index.tolist()
            self.test_indices = test_df.index.tolist()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.loc[idx, 'Image Index']
        label = self.df.loc[idx, 'label']
        img_path = self._find_image(img_name)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label, img_name

    def _find_image(self, filename):
        for i in range(1, 13):
            path = os.path.join(self.root_dir, f"images_{str(i).zfill(3)}_lighter", "images", filename)
            if os.path.exists(path):
                return path
        raise FileNotFoundError(f"{filename} not found.")

    
    def get_dataloader(self, from_split='train', indices=None, sample_size=None, batch_size=32, shuffle=False):
        """
        Create a PyTorch DataLoader for a specific subset of the dataset.

        This method provides flexible data loading for active learning scenarios where
        you need to dynamically select which samples to include.

        Args:
            from_split (str): Which split to use
                - 'train': Use training set
                - 'test': Use test set
                - Other: Use entire dataset
            indices (list or set, optional): Specific image indices to include.
                If provided, only these samples are included regardless of from_split.
            sample_size (int, optional): Number of samples to randomly select from the split.
                Only applies when from_split='train' and indices=None.
            batch_size (int): Batch size for DataLoader (default: 32)
            shuffle (bool): Whether to shuffle the data (default: False)

        Returns:
            DataLoader: PyTorch DataLoader with the selected subset

        Example:
            >>> # Get full training DataLoader
            >>> train_loader = dataset.get_dataloader(from_split='train', batch_size=32, shuffle=True)
            >>>
            >>> # Get DataLoader for specific indices (active learning)
            >>> selected_indices = ['image_001.jpg', 'image_002.jpg']
            >>> subset_loader = dataset.get_dataloader(indices=selected_indices, batch_size=16)
        """
        if from_split == 'train':
            df = self.train_df
            if sample_size is not None:
                df = df.sample(sample_size, random_state=42)
        elif from_split == 'test':
            df = self.test_df
        else:
            df = self.df
        if indices is not None:
            if not isinstance(indices, list):
                indices = list(indices)
            df = df.loc[indices]
        dataset = ChestXrayDatasetTrain(df, self.root_dir)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

# dataset_path = "nih_chest_xrays_light"
# dataset = ChestXrayDataset(dataset_path, split_type='from_files')
# print(dataset.train_df.head())