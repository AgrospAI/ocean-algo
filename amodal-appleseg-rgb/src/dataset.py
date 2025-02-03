import re
import os
import cv2
import json
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.transforms import Compose
from numpy.typing import NDArray
from typing import Tuple
from torch.utils.data import Dataset


class AppleSegmentationDataset(Dataset):
    def __init__(self, images_root: str, type: str, transform: Compose = None):
        self.root = images_root
        self.transform = transform
        self.annotations = self.load_annotations(type)        
        self.images = os.listdir(images_root)

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx: int):
        image_name = self.images[idx]
        image_path = os.path.join(self.root, image_name)
        image = Image.open(image_path)

        # generate segmentation mask for that image
        mask = self.get_mask(image_name, image.size[::-1])

        if self.transform:
            image = self.transform(image)
        
        mask = mask.unsqueeze(0) # mask (H, W) => (1, H, W)
        mask = transforms.functional.resize(mask, (256, 256))

        return image, mask
    
    def get_mask(self, image_name: str, image_size: Tuple[int, int]):
        pattern = re.compile(rf'^{re.escape(image_name)}\d*$')

        match = next((k for k in self.annotations.keys() if pattern.match(k)), None)

        mask = self.add_region_to_mask(image_size, match)

        return mask

    def add_region_to_mask(self, image_size: Tuple[int, int], match: str):
        #print(match)
        mask = np.zeros(image_size, dtype=np.uint8)

        apple_regions = max(map(int, list(self.annotations[match]['regions'].keys()))) if self.annotations[match]['regions'] else -1
        for region in range(apple_regions + 1):
            str_region = str(region)
            
            if str_region in self.annotations[match]['regions']:
                x_coords = np.array(self.annotations[match]['regions'][str_region]['shape_attributes']['all_points_x'])     
                y_coords = np.array(self.annotations[match]['regions'][str_region]['shape_attributes']['all_points_y'])
            else:
                x_coords, y_coords = np.array([]), np.array([])
            
            self.append_to_mask(x_coords, y_coords, mask)
        return torch.tensor(mask)

    @staticmethod
    def load_annotations(type: str):
        path = f'../../02-annotated_data_fuji/gt_json/train/via_region_data_{type}.json'
            
        with open(path, 'r') as annotations:
            return json.load(annotations)

    @staticmethod
    def append_to_mask(x_coords: NDArray[np.int64], y_coords: NDArray[np.int64], mask: NDArray[np.uint8]):
        if len(x_coords) > 0 and len(y_coords) > 0:
            points = np.array(list(zip(x_coords, y_coords)), dtype=np.int32)
            cv2.fillPoly(mask, [points], 1)
        return torch.tensor(mask, dtype=torch.float32)
        