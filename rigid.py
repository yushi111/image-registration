from audioop import add
from dis import dis
from utils import *
from model import *
import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt


DEIVICE='cpu'
if torch.cuda.is_available():
    DEIVICE='cuda'
    print('cuda is avaliable!')
data=get_data()
data=linear_mapping(data)

filtered=np.copy(data)
N,W,D,T=data.shape
#for t in range(T):
#    filtered[:,:,:,t]=low_pass_3d(data[:,:,:,t],sigma=5)


#print(data[:,:,20,15],filtered[:,:,20,15])
data=torch.tensor(data,dtype=torch.float64).to(DEIVICE)
#filtered=torch.tensor(filtered,dtype=torch.float64).to(DEIVICE)
#ref_img=get_ref(filtered.cpu()).to(DEIVICE)
#show_img(ref_img[:,:,20].cpu(),additional_data=ref_img[:,:,30].cpu(),name='ref.jpg')
EPOCH=1600
Alpha=0.05
lr=3e-6
for t in tqdm(range(T)):
    rigid_matrix=torch.randn((4,4),dtype=torch.double,device=DEIVICE,requires_grad=True)*10
    rigid_matrix=nn.init.kaiming_uniform(rigid_matrix)
    mask=torch.tensor([[1,0,0,0],
                    [0,1,0,0],
                    [0,0,1,0],
                    [1,1,1,1]],dtype=torch.bool,device=DEIVICE)
    coor=get_3d_coordinate(N,W,D,device=DEIVICE)
    with torch.no_grad():
        rigid_matrix[~mask]=0
    mask=mask.to(torch.double)
    
    for epoch in range(EPOCH):
        coor_add1=torch.cat((coor,torch.ones((N,W,D,1),dtype=torch.double,device=DEIVICE)),dim=3)
        newcoor=torch.mm(coor_add1.view(-1,4),rigid_matrix).view((N,W,D,4))
        newcoor=newcoor[:,:,:,0:3]
        outimg=sampler(data[:,:,:,t],newcoor-coor,devices=DEIVICE)
        loss=torch.sum(torch.abs(outimg-data[:,:,:,0]))
        grads=torch.autograd.grad(loss,rigid_matrix)[0]
        grads*=mask
        rigid_matrix=rigid_matrix-lr*grads
        print(rigid_matrix)
        print(f'epoch {epoch}: {loss}')
        if epoch%200==0:
            print(grads)
            show_img(data=data[:,:,20,t].cpu().detach(),additional_data=outimg[:,:,20].cpu().detach(),name=f'rigid{epoch}time{t}')
