from torch.utils.data import DataLoader, Dataset
from skimage import io, transform
import matplotlib.pyplot as plt
import os
import pytorch
from torchvision import transforms
import numpy as np

class AnimalData(Dataset):
    def __init__(selfself, root_dir, transform = None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = os.listdir(self.root_dir)
    def __len__(self):
        return len(self.images)
    def __getitem__(self, index):
        image_index = self.images[index]
        img_path = os.path.join(self.root_dir, image_index)
        img = io.imread(img_path)
        label = img_path.split('\\')[-1].split('.')[0]