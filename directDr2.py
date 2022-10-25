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
import argparse
import os
import re

parser=argparse.ArgumentParser(description="Motion correction tool")
parser.add_argument("--sigma",default=8,type=float)
parser.add_argument("--epoch",default=1600,type=int)
parser.add_argument("--alpha",default=5,type=float)
parser.add_argument("--lr",default=0.03,type=float)
parser.add_argument("--beta",default=0.01,type=float)
args=parser.parse_args()

DEIVICE='cpu'
if torch.cuda.is_available():
    DEIVICE='cuda'
    print('cuda is avaliable!')
for file in os.listdir('/ll4/data/4/animal/20220714_awakeGIMRI/processed/'):
    if re.match('.*gz',file):
        print(file)
        data=get_nii_data(os.path.join('/ll4/data/4/animal/20220714_awakeGIMRI/processed/',file))
        #data=get_data()
        foldername=os.path.splitext(os.path.splitext(file)[0])[0]
        pathname=os.path.join('20220714_awakeGIMRI',foldername)
        if not os.path.exists(pathname):
            os.makedirs(pathname)
        
        data=linear_mapping(data)

        filtered=np.copy(data)
        N,W,D,T=data.shape
        print(args.sigma)
        #for t in range(T):
        #    filtered[:,:,:,t]=low_pass_3d(data[:,:,:,t],sigma=args.sigma)

        for t in range(T):
            filtered[:,:,:,t]=low_pass_with_cutoff(data[:,:,:,t],42,52)
        show_img(filtered[:,:,10,10],additional_data=data[:,:,10,10],name='filter5060_1.png')
        show_img(filtered[:,:,12,10],additional_data=data[:,:,12,10],name='filter5060_2.png')

        init_coor=get_3d_coordinate(N,W,D)
        #print(data[:,:,20,15],filtered[:,:,20,15])
        data=torch.tensor(data,dtype=torch.float64).to(DEIVICE)
        filtered=torch.tensor(filtered,dtype=torch.float64).to(DEIVICE)
        #ref_img=get_ref(filtered.cpu()).to(DEIVICE)
        #ref_img=filtered[:,:,:,5]
        #show_img(ref_img[:,:,20].cpu(),additional_data=ref_img[:,:,30].cpu(),name='ref.jpg')
        Display_slice=12
        EPOCH=args.epoch
        print(EPOCH)
        Alpha=args.alpha
        Beta=args.beta
        IMGNORM=N*W*D
        Ref_width=7
        #fix here time is not correct
        if Ref_width%2==0:
            Ref_width+=1
        for t in tqdm(range(T)):
            offset=Ref_width//2
            if t<offset:
                ref_img=torch.mean(filtered[:,:,:,0:t+offset],dim=3)
            elif t>T-offset:
                ref_img=torch.mean(filtered[:,:,:,t-offset:T],dim=3)
            else:
                ref_img=torch.mean(filtered[:,:,:,t-offset:t+offset],dim=3)
            show_img(ref_img[:,:,10].cpu().detach(),name=f"{pathname}/ref{t}.png")
            dr=torch.randn((N,W,D,3),dtype=torch.double,device=DEIVICE,requires_grad=True)
            dr=nn.init.kaiming_uniform(dr)
            optimizer=optim.Adam([dr],args.lr)
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
                loss=(torch.sum(torch.abs(out-ref_img))+divpen*Alpha)
                #if prev_out is None:
                #    loss=(torch.sum(torch.abs(out-ref_img))+divpen*Alpha)/IMGNORM
                #else:
                #    loss=(torch.sum(torch.abs(out-ref_img))+divpen*Alpha+torch.sum(out-prev_out)*Beta)/IMGNORM
                loss.backward()
                prev_out=out.detach()
                if epoch%10==0:
                    loss_history.append(loss.cpu().detach())
                    div_history.append(Alpha*divpen.cpu().detach())
                    stepx.append(epoch)
                    if len(loss_history)>80:
                        loss_history.pop(0)
                        div_history.pop(0)
                        stepx.pop(0)
                    print(f"{epoch}: {loss}")
                    print(f"{epoch} divpen: {divpen*Alpha}")
                optimizer.step()
                #scheduler.step(loss)
                
                real=sampler(data=data[:,:,:,t].detach(),displacement=dr.detach(),devices=DEIVICE)
                if epoch%500==499:
                    #plot_3dvector_field(init_coor,dr.cpu().detach(),name=f'vectorf{t}_{epoch}.svg')
                    show_img(data=real[:,:,Display_slice].cpu(),name=f"{pathname}/epoch{t} real {epoch}.png")
                    show_img(data=data[:,:,Display_slice,t].cpu(),name=f"{pathname}/epoch{t} processed {epoch}.png")
                    show_img(real[:,:,Display_slice].cpu(),additional_data=(data[:,:,Display_slice,t]-real[:,:,Display_slice]).cpu(),name=f"{pathname}/diff{t} {epoch}.png")
                    #plot_loss(stepx,loss_history,epoch,t,div=div_history)
                    #for para in optimizer.param_groups:
                    #    print(para['lr'])
                    #    if para['lr']<1e-5:
                    #        newopoch=True
                #if newopoch:
                #    break
            real=sampler(data=data[:,:,:,t].detach(),displacement=dr.detach(),devices=DEIVICE)
            torch.save(real,f"{pathname}/processed{t}.pt")
            #torch.save(dr.cpu().detach(),f"dr{t}.pt")
