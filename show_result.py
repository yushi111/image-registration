import os
import torch 
from utils import *
from tqdm import tqdm

with open("processed.npy",'rb') as f:
    a=np.load(f)
    _,_,S,T=a.shape
    for t in tqdm(range(T)):
        show_img(a[:,:,:,t],name=f"oct13_processed_time{t}.png",argv=[i for i in range(S)])

data=get_nii_data()
#data=get_data()
data=linear_mapping(data)

filtered=np.copy(data)
N,W,D,T=data.shape
#for t in range(T):
#    filtered[:,:,:,t]=low_pass_3d(data[:,:,:,t],sigma=args.sigma)

for t in range(T):
    filtered[:,:,:,t]=low_pass_with_cutoff(data[:,:,:,t],42,52)

Ref_width=7
for t in tqdm(range(T)):
    show_img(data[:,:,:,t],name=f"oct13_origin_time{t}.png",argv=[i for i in range(S)])
    offset=Ref_width//2
    if t<offset:
        ref_img=np.mean(filtered[:,:,:,0:t+offset],axis=3)
    elif t>T-offset:
        ref_img=np.mean(filtered[:,:,:,t-offset:T],axis=3)
    else:
        ref_img=np.mean(filtered[:,:,:,t-offset:t+offset],axis=3)
    show_img(ref_img,name=f"oct13_ref_time{t}.png",argv=[i for i in range(S)])
        
