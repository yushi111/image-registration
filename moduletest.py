from dis import dis
from utils import *
from model import *
import torch
from torch.autograd import Variable
import torch.optim as optim
data=get_data()

filtered=low_pass_3d(data,sigma=3)


#print(data[:,:,20,15],filtered[:,:,20,15])
data=torch.tensor(data,dtype=torch.float64)
N,W,D,T=data.shape
#displacement=torch.zeros((N,W,D,3),dtype=torch.float64)
#displacement[:,:,:,2]=displacement[:,:,:,2]+40
#out=sampler(data=data[:,:,:,15], displacement=displacement)
#print(torch.sum(torch.abs(data[:,:,:,15]-out)))
#show_img(data=data[:,:,20,15],additional_data=out[:,:,20])

net=DisplaceNet().double()
xaxis=torch.linspace(0,N-1,steps=N,dtype=torch.float64)
yaxis=torch.linspace(0,W-1,steps=W,dtype=torch.float64)
zaxis=torch.linspace(0,D-1,steps=D,dtype=torch.float64)
xx,yy,zz=torch.meshgrid(xaxis,yaxis,zaxis)
initial_crd=torch.stack((zz,yy,xx),dim=3)
initial_crd=Variable(initial_crd,requires_grad=True)

ref_img=get_ref(data)
optimizer=optim.Adam(net.parameters(),lr=0.1,betas=(0.99,0.95))
EPOCH=10
for epoch in range(EPOCH):
    optimizer.zero_grad()
    displacement=net(initial_crd)
    out=sampler(data=data[:,:,:,15],displacement=displacement)
    loss=torch.sum(torch.abs(out-ref_img[:,:,:]))
    loss.backward()
    optimizer.step()
    print(epoch,loss)
    print(displacement)
    show_img(data=ref_img[:,:,20],additional_data=out.detach()[:,:,15],name=f"epoch {epoch}")
