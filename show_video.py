import os
import torch 
from utils import *
from tqdm import tqdm

#data=torch.load('processed69.pt')
data=torch.load('processed0.pt',map_location=torch.device('cpu'))
data=data.unsqueeze(3)
for i in tqdm(range(28)):
    newdata=torch.load(f'processed{i}.pt',map_location=torch.device('cpu')).unsqueeze(3)
    data=torch.cat((data,newdata),3)

create_video(data.cpu(),slices=[15,20,25,30],name="animal_processed.gif")
#create_video(data.cpu(),slices=[15],name="alpha008lr01drop03a5withdiff15.gif")
#create_video(data.cpu(),slices=[25],name="alpha008lr01drop03a5withdiff25.gif")
#create_video(data.cpu(),slices=[30],name="alpha008lr01drop03a5withdiff30.gif")
#create_video(data.cpu(),slices=[40],name="alpha008lr01drop03a5withdiff40.gif")
