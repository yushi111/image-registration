from utils import *
import torch 

data=get_nii_data()
create_video(data,[i for i in range(24)],'animal.gif')