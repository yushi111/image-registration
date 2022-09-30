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
for t in range(T):
    filtered[:,:,:,t]=low_pass_3d(data[:,:,:,t],sigma=5)


#print(data[:,:,20,15],filtered[:,:,20,15])
data=torch.tensor(data,dtype=torch.float64).to(DEIVICE)
filtered=torch.tensor(filtered,dtype=torch.float64).to(DEIVICE)
ref_img=get_ref(filtered.cpu()).to(DEIVICE)
#show_img(ref_img[:,:,20].cpu(),additional_data=ref_img[:,:,30].cpu(),name='ref.jpg')
EPOCH=1600
Alpha=0.05
#fix here time is not correct
for t in tqdm(range(T)):
    dr=torch.randn((N,W,D,3),dtype=torch.double,device=DEIVICE,requires_grad=True)
    dr=nn.init.kaiming_uniform(dr)
    optimizer=optim.Adam([dr],0.04)
    loss_history=[]
    stepx=[]
    for epoch in range(EPOCH):
        optimizer.zero_grad()
        out=sampler(data=filtered[:,:,:,t],displacement=dr,devices=DEIVICE)
        divpen=divergence(dr)
        loss=torch.sum(torch.abs(out-ref_img))+divpen*Alpha
        loss.backward()

        if epoch%10==0:
            loss_history.append(loss.cpu().detach())
            stepx.append(epoch)
            print(f"{epoch}: {loss}")
            print(f"{epoch} divpen: {divpen*Alpha}")
        optimizer.step()
        real=sampler(data=data[:,:,:,t].detach(),displacement=dr.detach(),devices=DEIVICE)
        if epoch%100==0:
            show_img(data=data[:,:,20,t].cpu(),additional_data=real[:,:,20].cpu(),name=f"epoch{t} {epoch}.png")
            show_img(real[:,:,20].cpu(),additional_data=(data[:,:,20,t]-real[:,:,20]).cpu(),name=f"diff{t} {epoch}.png")
            plot_loss(stepx,loss_history,epoch,t)
    real=sampler(data=data[:,:,:,t].detach(),displacement=dr.detach(),devices=DEIVICE)
    torch.save(real,f"processed{t}.pt")
