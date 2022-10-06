from audioop import add
from dis import dis
from mimetypes import init
from utils import *
from model import *
import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import *


DEIVICE='cpu'
if torch.cuda.is_available():
    DEIVICE='cuda'
    print('cuda is avaliable!')
#data=get_nii_data()
data=get_data()
data=linear_mapping(data)

filtered=np.copy(data)
N,W,D,T=data.shape
for t in range(T):
    filtered[:,:,:,t]=low_pass_3d(data[:,:,:,t],sigma=8)

init_coor=get_3d_coordinate(N,W,D)
#print(data[:,:,20,15],filtered[:,:,20,15])
data=torch.tensor(data,dtype=torch.float64).to(DEIVICE)
filtered=torch.tensor(filtered,dtype=torch.float64).to(DEIVICE)
#ref_img=get_ref(filtered.cpu()).to(DEIVICE)
ref_img=filtered[:,:,:,5]
#show_img(ref_img[:,:,20].cpu(),additional_data=ref_img[:,:,30].cpu(),name='ref.jpg')
Display_slice=20
EPOCH=1200
Alpha=5
Beta=0.01
IMGNORM=N*W*D
#fix here time is not correct
for t in tqdm(range(T)):
    dr=torch.randn((N,W,D,3),dtype=torch.double,device=DEIVICE,requires_grad=True)
    dr=nn.init.kaiming_uniform(dr)
    optimizer=optim.Adam([dr],0.03)
    #scheduler=ReduceLROnPlateau(optimizer,factor=0.7,patience=50,threshold=1)
    loss_history=[]
    stepx=[]
    div_history=[]
    prev_out=None
    newopoch=False
    for epoch in range(EPOCH):
        
        optimizer.zero_grad()
        out=sampler(data=filtered[:,:,:,t],displacement=dr,devices=DEIVICE)
        
        divpen=divergence(dr)
        real=sampler(data=data[:,:,:,t].detach(),displacement=dr,devices=DEIVICE)
        if prev_out is None:
            loss=(torch.sum(torch.abs(out-ref_img))+divpen*Alpha)/IMGNORM
        else:
            loss=(torch.sum(torch.abs(out-ref_img))+divpen*Alpha+torch.sum(out-prev_out)*Beta)/IMGNORM
        loss.backward()
        prev_out=out.detach()
        if epoch%10==0:
            loss_history.append(loss.cpu().detach()/IMGNORM)
            div_history.append(Alpha*divpen.cpu().detach()/IMGNORM)
            stepx.append(epoch)
            if len(loss_history)>80:
                loss_history.pop(0)
                div_history.pop(0)
                stepx.pop(0)
            print(f"{epoch}: {loss}")
            print(f"{epoch} divpen: {divpen*Alpha/IMGNORM}")
        optimizer.step()
        #scheduler.step(loss)
        
        real=sampler(data=data[:,:,:,t].detach(),displacement=dr.detach(),devices=DEIVICE)
        if epoch%500==0:
            plot_3dvector_field(init_coor,dr.cpu().detach(),name=f'vectorf{t}_{epoch}.svg')
            show_img(data=data[:,:,Display_slice,t].cpu(),additional_data=real[:,:,Display_slice].cpu(),name=f"epoch{t} {epoch}.png")
            show_img(real[:,:,Display_slice].cpu(),additional_data=(data[:,:,Display_slice,t]-real[:,:,Display_slice]).cpu(),name=f"diff{t} {epoch}.png")
            plot_loss(stepx,loss_history,epoch,t,div=div_history)
            #for para in optimizer.param_groups:
            #    print(para['lr'])
            #    if para['lr']<1e-5:
            #        newopoch=True
        #if newopoch:
        #    break
    real=sampler(data=data[:,:,:,t].detach(),displacement=dr.detach(),devices=DEIVICE)
    torch.save(real,f"processed{t}.pt")
