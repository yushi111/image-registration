import os
import torch 
from utils import *
from tqdm import tqdm

#data=torch.load('processed69.pt')
foldername='20220721_awakeGIMRI'
filename='file1410'
original=get_nii_data(f'/ll4/data/4/animal/{foldername}/processed/{filename}.nii.gz')
data=torch.load(f'{foldername}/{filename}/processed0.pt',map_location=torch.device('cpu'))
data=data.unsqueeze(3)
for i in tqdm(range(300)):
    newdata=torch.load(f'{foldername}/{filename}/processed{i}.pt',map_location=torch.device('cpu')).unsqueeze(3)
    data=torch.cat((data,newdata),3)

create_video(data.cpu(),slices=[i for i in range(24)],name=f"{foldername}/{filename}/animal_processed_4252w7.gif")
np.save(f"{foldername}/{filename}/processed.npy",data.numpy())
create_video(original,slices=[i for i in range(24)],name=f"{foldername}/{filename}/original.gif")
#create_video(data.cpu(),slices=[i*3+5 for i in range(12)],name="alpha20lr003a5.gif")
#create_video(data.cpu(),slices=[25],name="alpha008lr01drop03a5withdiff25.gif")
#create_video(data.cpu(),slices=[30],name="alpha008lr01drop03a5withdiff30.gif")
#create_video(data.cpu(),slices=[40],name="alpha008lr01drop03a5withdiff40.gif")
