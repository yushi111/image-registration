from utils import *
import torch

data=get_data()

filtered=low_pass_3d(data,sigma=3)


#print(data[:,:,20,15],filtered[:,:,20,15])
data=torch.tensor(data,dtype=torch.float64)
N,W,D,T=data.shape
displacement=torch.zeros((N,W,D,3),dtype=torch.float64)
displacement[:,:,:,2]=displacement[:,:,:,2]+40
out=sampler(data=data[:,:,:,15], displacement=displacement)
print(torch.sum(torch.abs(data[:,:,:,15]-out)))
show_img(data=data[:,:,20,15],additional_data=out[:,:,20])