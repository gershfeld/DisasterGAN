import glob
import random
import os
import numpy as np

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

disaster_type_dict = {"volcano": 0, "flooding": 1, "fire": 2}

class DisasterDataset(Dataset):
    def __init__(self, root_dir, transforms=None):
        self.transform = transforms
        self.root_dir = root_dir
        self.image_files = [f for f in os.listdir(root_dir) if not f.startswith('.')]
        # self.pre_dir = os.path.join(root, 'pre')
        # self.post_dir = os.path.join(root, 'post')
        # self.samples = os.listdir(self.pre_dir)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        filename = self.image_files[index]
        location, name = filename.split('-')
        disaster_type, id, post_or_pre, _ = name.split('_')

        pre_dis_file = os.path.join(self.root_dir, f"{location}-{disaster_type}_{id}_pre_disaster.png")
        post_dis_file = os.path.join(self.root_dir, f"{location}-{disaster_type}_{id}_post_disaster.png")

        pre_img = Image.open(pre_dis_file).convert('RGB')
        post_img = Image.open(post_dis_file).convert('RGB')

        if self.transform:
            pre_img = self.transform(pre_img)
            post_img = self.transform(post_img)

        return pre_img, post_img, disaster_type_dict[disaster_type]
