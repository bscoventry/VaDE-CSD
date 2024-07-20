#-----------------------------------------------------------------------------------------------------------------------------
# Author: Brandon S Coventry, PhD           Wisconsin Institute for Translational Neuroengineering
# Date: 07/20/2024
# Purpose: Load and plot VaDE model responses
#-----------------------------------------------------------------------------------------------------------------------------
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

N_CLASSES = 10
PLOT_NUM_PER_CLASS = 50


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

def test(model, data_loader, device, plot_points):
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
        for t in range(8):
            points = zs_tsne[ts == t]
            ax.scatter(points[:, 0], points[:, 1], color=cmap(t), label=str(t))
        ax.legend()
        plt.show()
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
    data = pd.read_pickle('CSDTrain.pkl')
    data = addLabelsToDF(data)
    dataset = CSDDataset(data,transform=transforms.ToTensor())
    data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=128, shuffle=True,
            num_workers=1, pin_memory=if_use_cuda)
    plot_points = {}
    for ck in range(8):
        CSDList = np.empty((PLOT_NUM_PER_CLASS,101,101),dtype=np.float64)
        curCSD = data.Sinks.loc[data['labels']==ck].values
        
        pltNum = 0
        if len(curCSD)>PLOT_NUM_PER_CLASS-1:
            pltNum = PLOT_NUM_PER_CLASS
        else:
            pltNum = len(curCSD)
        for bc in range(pltNum):
            CSDList[bc,:,:] = curCSD[bc]
        curCSD = torch.from_numpy(CSDList)
        curCSD = curCSD.view(-1, 10201).to(device)
        plot_points[ck] = curCSD
    xs = []
    ts = []
    for t, x in plot_points.items():
        xs.append(x)
        t = torch.full((x.size(0),), t, dtype=torch.long)
        ts.append(t)
    plot_points = (torch.cat(xs, dim=0), torch.cat(ts, dim=0))
    model = VaDE(N_CLASSES, 10201, 10)
    model.load_state_dict(torch.load('VadE.pth'))
    model = model.to(device)
    #model.eval()
    test(model, data_loader, device, plot_points)

if __name__ == '__main__':
    main()
pdb.set_trace() 