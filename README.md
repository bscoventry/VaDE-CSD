# Variational Deep Embedding for 2D Current Source Density Analysis

This repository is an adaptation of variational deep embedding emplemented in Pytorch from https://github.com/mori97/VaDE which is an adaptation from the original Keras implementation (https://arxiv.org/pdf/1611.05148.pdf). The goal of this model is unsupervised clustering of 2D current source densities from uniform arrays situated in a cortical layer. To run:

'''
$ python main.py --epochs 300 --gpu 0 --pretrain parameters.pth
'''

To observe results:

'''
$ tensorboard --logdir runs
'''
