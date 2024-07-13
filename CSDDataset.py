import argparse
import numpy as np
import matplotlib.pyplot as plt
from munkres import Munkres
from sklearn.manifold import TSNE
import torch
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import os
from vade import VaDE, lossfun
import pandas as pd
import pdb
from torch.utils.data import Dataset
from PIL import Image
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

class CSDDataset(Dataset):
            def __init__(self, df, transform=None, target_transform=None):
                self.df = df
                self.root_dir = os.getcwd()
                self.transform = transform
                self.target_transform = target_transform
                self.data = self.df.Sinks
                self.targets = self.df.labels.values
            def __len__(self):
                return len(self.data)

            # def __getitem__(self, idx):
            #     #img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
            #     #image = read_image(img_path)
            #     if torch.is_tensor(idx):
            #         idx = idx.tolist()
            #     label = int(self.df.labels.iloc[idx])
            #     image = self.df.Sinks.iloc[idx]
            #     Cmin = np.min(image)
            #     Cmax = np.max(image)
            #     image = (image-Cmin)/(Cmax-Cmin)
            #     image = torch.Tensor(image).reshape(1, 101, 101)
            #     if self.transform:
            #         image = self.transform(image)
            #     if self.target_transform:
            #         label = self.target_transform(label)
            #     return image, label
            def __getitem__(self, index: int) -> Tuple[Any, Any]:
                """
                Args:
                    index (int): Index

                Returns:
                    tuple: (image, target) where target is index of the target class.
                """
                img, target = self.data[index], int(self.targets[index])
                
                Cmin = np.min(img)
                Cmax = np.max(img)
                img = (img-Cmin)/(Cmax-Cmin)
                # doing this so that it is consistent with all other datasets
                # to return a PIL Image
                img = Image.fromarray(img, mode="L")

                if self.transform is not None:
                    img = self.transform(img)

                return img, target