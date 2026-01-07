"""Custom faces dataset."""
import os

import torch
from PIL import Image
from torch.utils.data import Dataset


class FacesDataset(Dataset):
    """Faces dataset.

    Attributes:
        root_path: str. Directory path to the dataset. This path has to
        contain a subdirectory of real images called 'real' and a subdirectory
        of not-real images (fake / synthetic images) called 'fake'.
        transform: torch.Transform. Transform or a bunch of transformed to be
        applied on every image.
    """
    def __init__(self, root_path: str, transform=None):
        """Initialize a faces dataset."""
        self.root_path = root_path
        self.real_image_names = os.listdir(os.path.join(self.root_path, 'real'))
        self.fake_image_names = os.listdir(os.path.join(self.root_path, 'fake'))
        self.transform = transform

    def __getitem__(self, index) -> tuple[torch.Tensor, int]:
        """Get a sample and label from the dataset."""
        n_real = len(self.real_image_names)
        n_fake = len(self.fake_image_names)

        if index < 0 or index >= n_real + n_fake:
            raise IndexError(f"Index {index} out of range for dataset of size {self.__len__()}")

        if index < n_real:
            img_name = self.real_image_names[index]
            label = 0
            img_path = os.path.join(self.root_path, 'real', img_name)
        else:
            img_name = self.fake_image_names[index - n_real]
            label = 1
            img_path = os.path.join(self.root_path, 'fake', img_name)

        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        """Return the number of images in the dataset."""
        return len(self.real_image_names) + len(self.fake_image_names)
