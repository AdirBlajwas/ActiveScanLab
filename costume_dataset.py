
from torch.utils.data import Dataset
from PIL import Image
import os

class ChestXrayDataset(Dataset):
    def __init__(self, df, root_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform

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
