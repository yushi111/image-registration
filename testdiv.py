from dis import dis
from utils import *
from model import *
import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm

x=torch.zeros((3,4,5,3))
x[:,:,:,0]=2
x[:,:,:,1]=5
x[:,:,:,2]=torch.randn((3,4,5))
print(divergence(x))