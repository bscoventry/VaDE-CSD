import os
import numpy as np
import torch
import torchvision
import argparse
from collections import OrderedDict
import pandas as pd
from modules import transform, resnet, network
from utils import yaml_config_hook
from torch.utils import data
import torch.utils.data.distributed
from evaluation import evaluation
from train import train_net
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms
from CSDDataset import CSDDataset
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

def main():
    parser = argparse.ArgumentParser()
    config = yaml_config_hook.yaml_config_hook("config/config.yaml")

    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    args = parser.parse_args()
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    # prepare data---------------------------------------------------------------------------------------------------------------------------------------------------
    #train data
    data = pd.read_pickle('CSDTrain.pkl')
    data = addLabelsToDF(data)
    train,test = train_test_split(data,test_size=0.1)
    train_dataset = CSDDataset(data,transform=transforms.ToTensor())
    # train_dataset = torchvision.datasets.CIFAR10(
    #         root=args.dataset_dir,
    #         download=True,
    #         train=True,
    #         transform=transform.Transforms(size=args.image_size, s=0.5),
    #     )
    # test_dataset = torchvision.datasets.CIFAR10(
    #         root=args.dataset_dir,
    #         download=True,
    #         train=False,
    #         transform=transform.Transforms(size=args.image_size, s=0.5),
    #     )
    #dataset = data.ConcatDataset([train_dataset, test_dataset])
    data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
                                              pin_memory=True)


    # test data
    # test_dataset_1 = torchvision.datasets.CIFAR10(
    #     root=args.dataset_dir,
    #     download=True,
    #     train=True,
    #     transform=transform.Transforms(size=args.image_size).test_transform,
    # )
    # test_dataset_2 = torchvision.datasets.CIFAR10(
    #     root=args.dataset_dir,
    #     download=True,
    #     train=False,
    #     transform=transform.Transforms(size=args.image_size).test_transform,
    # )
    dataset_test = CSDDataset(test,transform=transforms.ToTensor())#data.ConcatDataset([test_dataset_1, test_dataset_2])
    test_loader = torch.utils.data.DataLoader(
        dataset=dataset_test,
        batch_size=args.test_batch_size,
        shuffle=False)


    # Initializing our network with a network trained with CC -------------------------------------------------------------------------------------------------------
    res = resnet.get_resnet(args.resnet)
    net = network.Network(res, args.feature_dim, args.class_num)
    net = net.to('cuda')
    checkpoint = torch.load('CIFAR_10_initial', map_location=torch.device('cuda:0'))
    new_state_dict = OrderedDict()
    for k, v in checkpoint['net'].items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict)

    # optimizer ---------------------------------------------------------------------------------------------------------------------------------------------
    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)

    # train loop ---------------------------------------------------------------------------------------------------------------------------------------------------
    for epoch in range(args.start_epoch, args.max_epochs):

        print("epoch:", epoch)
        evaluation.net_evaluation(net,test_loader,args.dataset_size, args.test_batch_size)
        net, optimizer = train_net(net, data_loader, optimizer, args.batch_size, args.zeta)

        state = {'net': net.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
        with open('CIFAR_10_C3_loss_epoch_{}'.format(epoch), 'wb') as out:
            torch.save(state, out)


if __name__ == "__main__":
    main()
