from dis import dis
from utils import *
from model import *
import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm



data=get_data()
data=linear_mapping(data)
filtered=linear_mapping(low_pass_3d(data,sigma=3))
filtered=torch.tensor(filtered,dtype=torch.float64).cuda()

#print(data[:,:,20,15],filtered[:,:,20,15])
data=torch.tensor(data,dtype=torch.float64).cuda()
N,W,D,T=data.shape
#displacement=torch.zeros((N,W,D,3),dtype=torch.float64)
#displacement[:,:,:,2]=displacement[:,:,:,2]+40
#out=sampler(data=data[:,:,:,15], displacement=displacement)
#print(torch.sum(torch.abs(data[:,:,:,15]-out)))
#show_img(data=data[:,:,20,15],additional_data=out[:,:,20])


xaxis=torch.linspace(0,N-1,steps=N,dtype=torch.float64)
yaxis=torch.linspace(0,W-1,steps=W,dtype=torch.float64)
zaxis=torch.linspace(0,D-1,steps=D,dtype=torch.float64)
xx,yy,zz=torch.meshgrid(xaxis,yaxis,zaxis)
initial_crd=torch.stack((zz,yy,xx),dim=3).cuda()
initial_crd=Variable(initial_crd,requires_grad=True)

ref_img=get_ref(data.cpu()).cuda()

EPOCH=20
#divergence
Alpha=0
#magnitude
Beta=1
output4d=None
for t in tqdm(range(T)):

    net=nn.DataParallel(DisplaceNet().double()).cuda()
    optimizer=optim.Adam(net.parameters(),lr=0.1)
    for epoch in range(EPOCH):
        optimizer.zero_grad()
        displacement=net(initial_crd)
        displacement=torch.clamp(displacement,-100,100)
        x,y,z=displacement[:,:,:,0].shape

        l1=torch.sum(displacement[:,:,:,0])
        l2=torch.sum(displacement[:,:,:,1])
        l3=torch.sum(displacement[:,:,:,2])

        divergence=0
        for i,l in enumerate([l1,l2,l3]):
            grad=torch.autograd.grad(l,initial_crd,retain_graph=True)
            divergence+=torch.sum(grad[0][:,:,:,i])

        #displacement.backward()
        print("divergence",divergence)
        print(torch.sum(torch.abs(displacement)))
        #print(initial_crd.grad)
        out=sampler(data=filtered[:,:,:,t],displacement=displacement,devices='cuda:0')
        loss=(torch.sum(torch.abs(out-ref_img[:,:,:]))+torch.abs(divergence)*Alpha+Beta*torch.sum(torch.abs(displacement)))/(N*W*D)
        loss.backward()
        optimizer.step()
        print(epoch,loss)
        #show_img(data=ref_img[:,:,20],additional_data=out.detach()[:,:,20]-filtered[:,:,20,15],name=f"contrast {epoch}")
        if epoch%5==0:
            show_img(data=ref_img[:,:,20].cpu(),additional_data=out.detach()[:,:,20].cpu(),name=f"epoch {epoch} T{t}")
            real=sampler(data=data[:,:,:,t],displacement=displacement,devices='cuda:0')
            show_img(data=data[:,:,20,t].cpu(),additional_data=real.detach()[:,:,20].cpu(),name=f"Real {epoch} T{t}")
            show_img(data=real.detach()[:,:,20].cpu(),additional_data=real.detach()[:,:,20].cpu()-data[:,:,20,t].cpu(),name=f"contrast {epoch} T{t}")
            #torch.save(net.state_dict(),'model.pt')
        if epoch==EPOCH-1:
            if output4d is None:
                output4d=real.detach().unsqueeze(3)
            else:
                output4d=torch.cat((output4d,real.detach().unsqueeze(3)),dim=3)
                torch.save(output4d.cpu(),f'processed{t}.pt')
                print(output4d.shape)

torch.save(output4d.cpu(),'processed.pt')
create_video(output4d.cpu())

    
    

