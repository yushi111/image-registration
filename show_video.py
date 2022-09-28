import os
import torch 
from utils import *

data=torch.load('processed69.pt')
create_video(data)
