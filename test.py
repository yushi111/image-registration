from utils import *
import torch 

displace=torch.load('dr1.pt')
N,W,D,_=displace.shape
coor=get_3d_coordinate(N,W,D)
plot_3dvector_field(coor,displace)