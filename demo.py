"""Main function of Graph Super-Resolution Network (GSR-Net) framework 
   for predicting high-resolution brain connectomes from low-resolution connectomes. 
    
    ---------------------------------------------------------------------
    
    This file contains the implementation of the training and testing process of our GSR-Net model.
        train(model, optimizer, subjects_adj, subjects_ground_truth, args)

                Inputs:
                        model:        constructor of our GSR-Net model:  model = GSRNet(ks,args)
                                      ks:   array that stores reduction rates of nodes in Graph U-Net pooling layers
                                      args: parsed command line arguments

                        optimizer:    constructor of our model's optimizer (borrowed from PyTorch)  

                        subjects_adj: (n × l x l) tensor stacking LR connectivity matrices of all training subjects
                                       n: the total number of subjects
                                       l: the dimensions of the LR connectivity matrices

                        subjects_ground_truth: (n × h x h) tensor stacking LR connectivity matrices of all training subjects
                                                n: the total number of subjects
                                                h: the dimensions of the LR connectivity matrices

                        args:          parsed command line arguments, to learn more about the arguments run: 
                                       python demo.py --help
                Output:
                        for each epoch, prints out the mean training MSE error


            
        test(model, test_adj,test_ground_truth,args)

                Inputs:
                        test_adj:      (n × l x l) tensor stacking LR connectivity matrices of all testing subjects
                                        n: the total number of subjects
                                        l: the dimensions of the LR connectivity matrices

                        test_ground_truth:      (n × h x h) tensor stacking LR connectivity matrices of all testing subjects
                                                 n: the total number of subjects
                                                 h: the dimensions of the LR connectivity matrices

                        see train method above for model and args.

                Outputs:
                        for each epoch, prints out the mean testing MSE error


    To evaluate our framework we used 5-fold cross-validation strategy.

    ---------------------------------------------------------------------
    Copyright 2020 Megi Isallari, Istanbul Technical University.
    All rights reserved.
    """


import torch
import numpy as np
import torch.optim as optim
from sklearn.model_selection import KFold
from preprocessing import *
from model import *
from train import *
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GSR-Net')
    parser.add_argument('--epochs', type=int, default=200, metavar='no_epochs',
                        help='number of episode to train ')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='lr',
                        help='learning rate (default: 0.0001 using Adam Optimizer)')
    parser.add_argument('--splits', type=int, default=5, metavar='n_splits',
                        help='no of cross validation folds')
    parser.add_argument('--lmbda', type=int, default=16, metavar='L',
                        help='self-reconstruction error hyperparameter')
    parser.add_argument('--lr_dim', type=int, default=160, metavar='N',
                        help='adjacency matrix input dimensions')
    parser.add_argument('--hr_dim', type=int, default=320, metavar='N',
                        help='super-resolved adjacency matrix output dimensions')
    parser.add_argument('--hidden_dim', type=int, default=320, metavar='N',
                        help='hidden GraphConvolutional layer dimensions')
    parser.add_argument('--padding', type=int, default=26, metavar='padding',
                        help='dimensions of padding')
    args = parser.parse_args()


# def train
# subjects_adj, subjects_ground_truth, test_adj, test_ground_truth = data()
# X = np.concatenate((subjects_adj, test_adj), axis=0)
# Y = np.concatenate((subjects_ground_truth, test_ground_truth), axis=0)

# SIMULATING THE DATA: EDIT TO ENTER YOUR OWN DATA
X = np.random.normal(0, 0.5, (277, 160, 160))
Y = np.random.normal(0, 0.5, (277, 320, 320))

cv = KFold(n_splits=args.splits, random_state=42, shuffle=False)
print("Torch: ")

# ks = [0]
ks = [0.9, 0.7, 0.6, 0.5]
model = GSRNet(ks, args)

# layer = ULayer()
print(model)
optimizer = optim.Adam(model.parameters(), lr=args.lr)

for train_index, test_index in cv.split(X):
    subjects_adj, test_adj, subjects_ground_truth, test_ground_truth = X[
        train_index], X[test_index], Y[train_index], Y[test_index]
    train(model, optimizer, subjects_adj, subjects_ground_truth, args)
    test(model, test_adj, test_ground_truth, args)
