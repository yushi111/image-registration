import os
import argparse
import time
from turtle import forward
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_uniform_(m.weight)

class DisplaceNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net1=nn.Sequential(
            nn.Conv3d(3,2,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv3d(2,1,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv3d(1,1,kernel_size=5,padding=2)
        )
        self.net2=nn.Sequential(
            nn.Conv3d(3,2,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv3d(2,1,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv3d(1,1,kernel_size=5,padding=2)
        )
        self.net3=nn.Sequential(
            nn.Conv3d(3,2,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv3d(2,1,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv3d(1,1,kernel_size=5,padding=2)
        )
        for layer in self.net1:
            weights_init(layer)
        for layer in self.net2:
            weights_init(layer)
        for layer in self.net3:
            weights_init(layer)
    
    def forward(self,input):
        """
        input: (W,H,D,3) cooridinates
        output: (W,H,D,1) displacement at one axis
        """
        input=input.unsqueeze(0)
        input=torch.permute(input,(0,4,1,2,3))
        dis_x=self.net1(input)
        dis_y=self.net2(input)
        dis_z=self.net3(input)
        dis_x=torch.permute(dis_x,(0,2,3,4,1)).squeeze()
        dis_y=torch.permute(dis_y,(0,2,3,4,1)).squeeze()
        dis_z=torch.permute(dis_z,(0,2,3,4,1)).squeeze()
        displacement=torch.stack((dis_z,dis_y,dis_x),dim=3)
        return displacement

