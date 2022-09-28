from dis import dis
from utils import *
from model import *
import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm

data=get_data()
data=torch.tensor(data)
N,W,D,T=data.shape
rotation_x=get_3d_rotation_matrix(20,'x').double()
rotation_y=get_3d_rotation_matrix(20,'y').double()
rotation_z=get_3d_rotation_matrix(20,'z').double()
print(rotation_z)
inic=get_3d_coordinate(N,W,D)
new_coordinate=torch.reshape(torch.mm(torch.reshape(inic,(-1,3)),rotation_x),(N,W,D,3))
#print(new_coordinate)
out=sampler(displacement=new_coordinate-inic,data=data[:,:,:,20])
show_img(data=data[:,:,20,20],additional_data=out[:,:,20],name='rotate20x.jpg')
new_coordinate=torch.reshape(torch.mm(torch.reshape(inic,(-1,3)),rotation_y),(N,W,D,3))
#print(new_coordinate)
out=sampler(displacement=new_coordinate-inic,data=data[:,:,:,20])
show_img(data=data[:,:,20,20],additional_data=out[:,:,20],name='rotate20y.jpg')
new_coordinate=torch.reshape(torch.mm(torch.reshape(inic,(-1,3)),rotation_z),(N,W,D,3))
#print(new_coordinate)
out=sampler(displacement=new_coordinate-inic,data=data[:,:,:,20])
show_img(data=data[:,:,20,20],additional_data=out[:,:,20],name='rotate20z.jpg')