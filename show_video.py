import os
import torch 
from utils import *
from tqdm import tqdm

#data=torch.load('processed69.pt')
data=torch.load('processed0.pt')
data=data.unsqueeze(3)
for i in tqdm(range(60)):
    newdata=torch.load(f'processed{i}.pt').unsqueeze(3)
    data=torch.cat((data,newdata),3)
create_video(data.cpu(),name="alpha005lr004.gif")
