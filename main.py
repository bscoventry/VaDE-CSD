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
from CSDDataset import CSDDataset
N_CLASSES = 8
PLOT_NUM_PER_CLASS = 128

def addLabelsToDF(df):
    EPP = df.EnergyPerPulse.values
    nRows = len(EPP)
    labels = np.zeros((nRows,),dtype=int)
    for ck in range(nRows):
        if EPP[ck] <0.5:
            labels[ck] = 0
        elif 0.5<= EPP[ck] < 1: 
            labels[ck] = 1
        elif 1<= EPP[ck] < 1.5: 
            labels[ck] = 2
        elif 1.5<= EPP[ck] < 2: 
            labels[ck] = 3
        elif 2<= EPP[ck] < 2.5: 
            labels[ck] = 4
        elif 2.5<= EPP[ck] < 3: 
            labels[ck] = 5
        elif 3<= EPP[ck] < 3.5: 
            labels[ck] = 6
        elif EPP[ck]>=3.5: 
            labels[ck] = 7
    df['labels'] = labels
    return df

# class CSDDataset(Dataset):
#     if __name__ == '__main__':
#         def __init__(self, df, transform=None, target_transform=None):
#             self.df = df
#             self.root_dir = os.getcwd()
#             self.transform = transform
#             self.target_transform = target_transform
#             self.data = self.df.Sinks
#             self.targets = self.df.labels.values
#         def __len__(self):
#             return len(self.data)

#         # def __getitem__(self, idx):
#         #     #img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
#         #     #image = read_image(img_path)
#         #     if torch.is_tensor(idx):
#         #         idx = idx.tolist()
#         #     label = int(self.df.labels.iloc[idx])
#         #     image = self.df.Sinks.iloc[idx]
#         #     Cmin = np.min(image)
#         #     Cmax = np.max(image)
#         #     image = (image-Cmin)/(Cmax-Cmin)
#         #     image = torch.Tensor(image).reshape(1, 101, 101)
#         #     if self.transform:
#         #         image = self.transform(image)
#         #     if self.target_transform:
#         #         label = self.target_transform(label)
#         #     return image, label
#         def __getitem__(self, index: int) -> Tuple[Any, Any]:
#             """
#             Args:
#                 index (int): Index

#             Returns:
#                 tuple: (image, target) where target is index of the target class.
#             """
#             img, target = self.data[index], int(self.targets[index])
            
#             Cmin = np.min(img)
#             Cmax = np.max(img)
#             img = (img-Cmin)/(Cmax-Cmin)
#             # doing this so that it is consistent with all other datasets
#             # to return a PIL Image
#             img = Image.fromarray(img, mode="L")

#             if self.transform is not None:
#                 img = self.transform(img)

#             return img, target

class CSD_Dataloader(torch.utils.data.Dataset):
    
        def __init__(self, CSDDF,source_or_sink):
            self.df = CSDDF
            if source_or_sink==1:
                self.CSD = self.df['Sinks']
            else:
                self.CSD = self.df['Sources']
        
            self.EPP = self.df['EnergyPerPulse']
        # get sample
        def __getitem__(self, idx):
            CSD_item = self.CSD[idx]
            EPP = self.EPP[idx]
            #scale data to 0 and 1
            Cmin = np.min(CSD_item)
            Cmax = np.max(CSD_item)
            CSD_item = (CSD_item-Cmin)/(Cmax-Cmin)
            
            # convert to tensor
            image = torch.Tensor(CSD_item).reshape(1, 101, 101)
            
            curEPP = self.df.labels[idx]
            target = curEPP

            return image, target

        def __len__(self):
            return len(self.images)


def train(model, data_loader, optimizer, device, epoch, writer):
    model.train()
    pdb.set_trace()
    total_loss = 0
    for x, _ in data_loader:
        
        x = x.to(device).view(-1, 10201)
        #x = x.to(device).view(-1, 784)
        recon_x, mu, logvar = model(x)
        loss = lossfun(model, x, recon_x, mu, logvar)
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    writer.add_scalar('Loss/train', total_loss / len(data_loader), epoch)


def test(model, data_loader, device, epoch, writer, plot_points):
    model.eval()

    gain = torch.zeros((N_CLASSES, N_CLASSES), dtype=torch.int, device=device)
    with torch.no_grad():
        for xs, ts in data_loader:
            xs, ts = xs.to(device).view(-1, 10201), ts.to(device)
            #xs, ts = xs.to(device).view(-1, 784), ts.to(device)
            ys = model.classify(xs)
            for t, y in zip(ts, ys):
                gain[t, y] += 1
        cost = (torch.max(gain) - gain).cpu().numpy()
        assign = Munkres().compute(cost)
        acc = torch.sum(gain[tuple(zip(*assign))]).float() / torch.sum(gain)

        # Plot latent space
        xs, ts = plot_points[0].to(device), plot_points[1].numpy()
        zs = model.encode(xs)[0].cpu().numpy()
        tsne = TSNE(n_components=2, init='pca')
        zs_tsne = tsne.fit_transform(zs)

        fig, ax = plt.subplots()
        cmap = plt.get_cmap("tab10")
        for t in range(10):
            points = zs_tsne[ts == t]
            ax.scatter(points[:, 0], points[:, 1], color=cmap(t), label=str(t))
        ax.legend()

    writer.add_scalar('Acc/test', acc.item(), epoch)
    writer.add_figure('LatentSpace', fig, epoch)


def main():
    parser = argparse.ArgumentParser(
        description='Train VaDE with MNIST dataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--epochs', '-e',
                        help='Number of epochs.',
                        type=int, default=100)
    parser.add_argument('--gpu', '-g',
                        help='GPU id. (Negative number indicates CPU)',
                        type=int, default=-1)
    parser.add_argument('--learning-rate', '-l',
                        help='Learning Rate.',
                        type=float, default=0.002)
    parser.add_argument('--batch-size', '-b',
                        help='Batch size.',
                        type=int, default=4)
    parser.add_argument('--pretrain', '-p',
                        help='Load parameters from pretrained model.',
                        type=str, default=None)
    args = parser.parse_args()

    if_use_cuda = torch.cuda.is_available() and args.gpu >= 0
    device = torch.device('cuda:{}'.format(args.gpu) if if_use_cuda else 'cpu')

    dataset = datasets.MNIST('./data', train=True, download=True,
                            transform=transforms.ToTensor())
    data = pd.read_pickle('CSDTrain.pkl')
    data = addLabelsToDF(data)
    dataset = CSDDataset(data,transform=transforms.ToTensor())
    #data_loader = CSD_Dataloader(data,source_or_sink=1)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=1, pin_memory=if_use_cuda)
    pdb.set_trace()
    # For plotting
    # plot_points = {}
    # for t in range(10):
    #     points = torch.cat([data for data, label in dataset if label == t])
    #     points = points.view(-1, 10201)[:PLOT_NUM_PER_CLASS].to(device)
    #     plot_points[t] = points
    # xs = []
    # ts = []
    # for t, x in plot_points.items():
    #     xs.append(x)
    #     t = torch.full((x.size(0),), t, dtype=torch.long)
    #     ts.append(t)
    # plot_points = (torch.cat(xs, dim=0), torch.cat(ts, dim=0))

    model = VaDE(N_CLASSES, 10201, 8)
    #model = VaDE(N_CLASSES, 784, 10)
    if args.pretrain:
        model.load_state_dict(torch.load(args.pretrain))
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    # LR decreases every 10 epochs with a decay rate of 0.9
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=10, gamma=0.9)

    # TensorBoard
    writer = SummaryWriter()

    for epoch in range(1, args.epochs + 1):
        train(model, data_loader, optimizer, device, epoch, writer)
        #test(model, data_loader, device, epoch, writer, plot_points)
        lr_scheduler.step()

    writer.close()
    pdb.set_trace()


if __name__ == '__main__':
    main()