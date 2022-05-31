import os

import cv2
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class FGVC_Dataset(Dataset):
    def __init__(self, data_path, is_train=True, is_test=False):
        self.data_path = data_path
        self.file_list = []
        label_map_file = os.path.join(data_path, 'variants.txt')
        with open(label_map_file, 'r') as f:
            classes = f.read().splitlines()
        
        self.label_map = {classes[i]:i for i in range(len(classes))}

        if is_train:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(size=(224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])
            file_reader = 'images_train.txt'

        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(size=(224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])
            if is_test:
                file_reader = 'images_test.txt'
            else: file_reader = 'images_val.txt'

        with open(os.path.join(data_path, file_reader)) as f:
            self.file_list = f.read().splitlines()
        
        
    def __len__(self):
        return len(self.file_list)

    def map_label(self, label):
        return self.label_map[label]

    def __getitem__(self, idx):
        image_name = self.file_list[idx]
        image_path = os.path.join(self.data_path, 'images', image_name + '.jpg')
        label_path = os.path.join(self.data_path, 'labels', image_name + '.txt')

        with open(label_path, 'r') as f:
            label = f.readline()
        
        label = torch.tensor(self.map_label(label))
        image = cv2.imread(image_path)
        image = self.transform(image)

        return image, label


if __name__ == '__main__':
    pass
