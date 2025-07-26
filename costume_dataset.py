from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from torchvision import transforms
import pandas as pd
from sklearn.model_selection import train_test_split

class ChestXrayDatasetTrain(Dataset):
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
        for i in range(1, 13):
            path = os.path.join(self.root_dir, f"images_{str(i).zfill(3)}_lighter", "images", filename)
            if os.path.exists(path):
                return path
        raise FileNotFoundError(f"{filename} not found.")




class ChestXrayDataset(Dataset):
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
            train_df, test_df = train_test_split(self.df, test_size=0.2, stratify=self.df['label'])
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

    
    def get_dataloader(self, from_split='train', indices=None, sample_size=None, batch_size=32):
        """
        """
        if from_split == 'train':
            df = self.train_df
            shuffle = True
            if sample_size is not None:
                df = df.sample(sample_size, random_state=42)
        elif from_split == 'test':
            df = self.test_df
            shuffle = False
        else:
            df = self.df
            shuffle = False
        if indices is not None:
            if not isinstance(indices, list):
                indices = list(indices)
            df = df.loc[indices]
        dataset = ChestXrayDatasetTrain(df, self.root_dir)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

# dataset_path = "nih_chest_xrays_light"
# dataset = ChestXrayDataset(dataset_path, split_type='from_files')
# print(dataset.train_df.head())