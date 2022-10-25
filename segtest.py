from utils import *
from Wnet import *
import scipy
from torch import optim

def div(data3d):
    divs=torch.gradient(data3d)
    out=0
    for div_v in divs:
        out+=torch.sum(torch.abs(div_v))

    return out
DEIVICE='cpu'
if torch.cuda.is_available():
    DEIVICE='cuda'
    print('cuda is avaliable!')


# tshape=[96,128,32]
# data=get_nii_data()
# N,W,D,T=data.shape
# #data=get_data()
# data=linear_mapping(data)
# data_reshape=np.zeros((96,128,32,T))
# for i,d in enumerate(tshape):
#     tshape[i]=d/data.shape[i]

# for i in range(data.shape[3]):
#     data_reshape[:,:,:,i]=scipy.ndimage.zoom(data[:,:,:,i],tshape)


# data_reshape=torch.tensor(data_reshape,dtype=torch.double).to(DEIVICE)
# torch.save(data_reshape,"data_reshape.pt")
data_reshape=torch.load("data_reshape.pt").unsqueeze(0).permute(4,0,1,2,3).to(DEIVICE)
_,_,N,W,D=data_reshape.shape
print(data_reshape.shape)
net=Net().to(DEIVICE)
batch_size=1
data_reshape=torch.split(data_reshape,batch_size,0)
assigned_color=torch.randn((5,),dtype=torch.double,device=DEIVICE,requires_grad=True)

EPOCH=100
optimizer=optim.Adam(net.parameters(),lr=5)
optimizer2=optim.Adam([assigned_color],lr=0.5)
for data in data_reshape:
    for epoch in tqdm(range(EPOCH)):
        optimizer.zero_grad()
        optimizer2.zero_grad()
        gummax_out,softmax_out=net(data)
        # loss=torch.sum(gummax_out)
        # loss.backward()
        # print(loss)
        #print(gummax_out.shape,softmax_out.shape)
        #print("diff:",torch.sum(torch.abs(torch.argmax(gummax_out,dim=1)-torch.argmax(softmax_out,dim=1))))
        #print(net.conv1[0].weight.grad)
        c=torch.argmax(softmax_out,dim=1)-softmax_out
        c=c.detach()
        pixel_label=c+softmax_out
        # gummax_out=gummax_out.to(torch.bool)
        # idx=torch.zeros_like(gummax_out,dtype=torch.long,device=DEIVICE)
        # for i in range(5):
        #    idx[:,i,:,:,:]=i
        # pixel_label=torch.argmax(softmax_out,dim=1)
        
        #pixel_label=idx[gummax_out].reshape((batch_size,N,W,D))
        #print("diff:",torch.sum(torch.argmax(gummax_out,dim=1)-torch.argmax(softmax_out,dim=1)))
        #print(pixel_label[0,:,4,1,2])
        pixel_label=pixel_label.sum(dim=1)/5
        #print(pixel_label)
        out_img=assigned_color[pixel_label.to(torch.long)]
        loss=torch.sum(torch.abs(out_img-data))/batch_size
        for b in range(batch_size):
            loss+=div(torch.flatten(pixel_label[b,:,:,:]))
        loss.backward()
        print(loss)
        
        
        optimizer2.step()
        if epoch%10==0:
            show_img(pixel_label[0,:,:,16].cpu().detach(),additional_data=out_img[0,:,:,16].cpu().detach(),name=f"test{epoch}.png")
       
        optimizer.step()
        #print(net.conv1[0].weight.grad)
        

