from utils import *
import torch 

data=get_data()
ref=get_ref(data)
show_img(data[:,:,20,20],additional_data=ref[:,:,20],name='ref.jpg')
N,W,D,T=data.shape
for t in range(T):
    data[:,:,:,t]=low_pass_3d(data[:,:,:,t],sigma=3)
print(data.shape)
#data_enhanced=linear_mapping(data)
#print(data_enhanced[:,98,20,15])
#show_img(data=data[:,:,20,15],additional_data=data_enhanced[:,:,20,15])
data=torch.tensor(data)
create_video(data,name='sigma3.gif')