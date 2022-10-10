import os
import torch 
from utils import *
from tqdm import tqdm
import numpy as np 

data=get_data()
data=linear_mapping(data)

#filtered=np.copy(data)
N,W,D,T=data.shape
cut_data=np.copy(data)
cut_data[cut_data<75]=0
show_hist(cut_data[:,:,:,20],name='hist_cut_70.png')
show_img(data[:,:,20,20],additional_data=cut_data[:,:,20,20],name='cut_at_70.jpg')
create_video(cut_data,slices=[i*2 for i in range(24)],name="cut70.gif")