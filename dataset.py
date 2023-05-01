import glob
import random
import os
import numpy as np

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

disaster_type_dict = {"volcano": 0, "flooding": 1, "fire": 2, "wildfire": 2, "tsunami": 3, "earthquake": 4,
                      "hurricane": 5}


class DisasterDataset(Dataset):
    def __init__(self, root_dir, transforms=None):
        self.transform = transforms
        self.root_dir = root_dir
        self.image_files = [f for f in os.listdir(root_dir) if not f.startswith('.')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        # extract the images name for finding the post and pre images
        filename = self.image_files[index]
        location, name = filename.rsplit('-', 1)
        disaster_type, id, post_or_pre, _ = name.split('_')

        pre_dis_file = os.path.join(self.root_dir, f"{location}-{disaster_type}_{id}_pre_disaster.png")
        post_dis_file = os.path.join(self.root_dir, f"{location}-{disaster_type}_{id}_post_disaster.png")

        pre_img = Image.open(pre_dis_file).convert('RGB')
        post_img = Image.open(post_dis_file).convert('RGB')

        if self.transform:
            pre_img = self.transform(pre_img)
            post_img = self.transform(post_img)
        # make sure files exists and disaster type exists
        if location in disaster_type_dict.keys():
            location, disaster_type = disaster_type, location
        return pre_img, post_img, disaster_type_dict[disaster_type]
