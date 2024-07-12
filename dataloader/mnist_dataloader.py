"""
this is simple dataloader for mnist dataloader.
this assumes that images are already extracted from csv
"""
import glob
import os

import cv2
import torch
from tqdm import tqdm
from torch.utils.data.dataset import Dataset


class MnistDataset(Dataset):
    def __init__(self, split, im_path, im_ext='png'):
        self.split = split
        self.im_ext = im_ext
        self.images, self.labels = self.load_images(im_path)

    def load_images(self, im_path):
        assert os.path.exists(im_path), f"image path {im_path} does not exists"

        ims = []
        labels = []

        for d_name in tqdm(os.listdir(im_path)):
            for f_name in glob.glob(os.path.join(im_path, d_name, '*.{}'.format(self.im_ext))):
                ims.append(f_name)
                labels.append(int(d_name))
        print(f'Found {len(ims)} images for split {self.split}')

        return ims, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        im = cv2.imread(self.images[index], 0)
        label = self.labels[index]

        # convert read image from 0 to 255 into -1 to 1
        im = 2 * (im/255) - 1

        # Convert H,W,C into 1,C,H,W
        im_tensor = torch.from_numpy(im)[None, :]
        #print(f'im_tensor shape is {im_tensor.shape}')

        return im_tensor, torch.as_tensor(label)



