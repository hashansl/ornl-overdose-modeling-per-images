
import random
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
import pandas as pd
import os
from skimage import io
import geopandas as gpd
import matplotlib.pyplot as plt
from skimage.transform import resize


import torchvision.transforms as transforms  # Transformations we can perform on our dataset
from torchvision.transforms import ToPILImage

from datasets import load_dataset

# Load the CIFAR-10 dataset
ds = load_dataset("uoft-cs/cifar10")

# # Define transform
# data_transform = transforms.Compose([
#     transforms.ToTensor()
# ])

# Prepare the data for PyTorch
class CIFAR10Dataset(torch.utils.data.Dataset):
    def __init__(self, split, transform=None):
        self.data = ds[split]
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]["img"]
        label = self.data[idx]["label"]

        # transform labels to tensor
        label = torch.tensor(label)

        # image = resize(image, (244, 244, 15), anti_aliasing=True)

        if self.transform:
            image = self.transform(image)
            
        return image, label



# dataset = CIFAR10Dataset("train", transform=data_transform)

# data = dataset[5000]

# print(data[0].shape)
# print(data[1])



# class data_loader_persistence_img(Dataset):

#     def __init__(self,annotation_file_path,root_dir,transform=None):
#         self.dtype = {'STCNTY': str}
#         self.annotations = pd.read_csv(annotation_file_path, dtype=self.dtype) # dtype defined here
#         # self.annotations = pd.read_csv(annotation_file_path)
#         self.root_dir = root_dir
#         self.transform = transform
#         self.class_names = sorted(self.annotations['OD_class_90'].unique())
#         self.to_pil = ToPILImage()  # Initialize ToPILImage transform

#     def __len__(self):
#         return len(self.annotations)

#     def __getitem__(self,index):

#         npy_file_path = os.path.join(self.root_dir, str(self.annotations.iloc[index,1]) + '.npy')
#         # npy_file_path = os.path.join(self.root_dir, str(self.annotations.iloc[index,0]) + '.npy')

#         img = np.load(npy_file_path).astype(np.float32)
#         img = resize(img, (244, 244, 15), anti_aliasing=True)

#         # img = self.to_pil(img)

#         y_label = torch.tensor(int(self.annotations.iloc[index]['OD_class_90']))

#         if self.transform:
#             img = self.transform(img)
#         return (img, y_label)

#     def get_class_names(self):
#         return self.class_names



# root_dir = "/home/h6x/git_projects/ornl-svi-data-processing/experiment_2/processed_data_1/npy_combined" # has 2
# annotation_file_path = "/home/h6x/git_projects/ornl-svi-data-processing/experiment_2/processed_data_1/annotations_2018_npy_2_classes_only_h0h1_90_percentile.csv"

# dataset = data_loader_persistence_img(annotation_file_path=annotation_file_path,root_dir=root_dir,transform=transforms.ToTensor())


# # get the number of labels in each class using dataset object
# class_names = dataset.get_class_names()

# print(class_names)
# print(len(class_names))






# print(len(dataset))
# # print(dataset.__getitem__(0)[0])
# print(dataset.__getitem__(2117)[1])


# print(len(dataset[0]))

# print(dataset[400][1])
# print(dataset[0][0].shape)

# train_set, test_set = torch.utils.data.random_split(dataset, [70, 25])


# #---
# train_data = DataLoader(dataset=train_set, batch_size=16, shuffle=True)
# test_data = DataLoader(dataset=test_set, batch_size=16, shuffle=False)

# class_names = dataset.get_class_names()
# print(class_names)







