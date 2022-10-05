import os
import torch 
from utils import *
from tqdm import tqdm

data=get_data()
data=linear_mapping(data)

filtered=np.copy(data)
N,W,D,T=data.shape
for t in range(T):
    filtered[:,:,:,t]=low_pass_3d(data[:,:,:,t],sigma=8)
ref_img=get_ref(filtered)
show_hist(ref_img)
show_hist(np.average(data,axis=3),name='hist_original.png')