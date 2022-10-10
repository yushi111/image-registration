from utils import *



data=get_data()
data=linear_mapping(data)

filtered=np.copy(data)
N,W,D,T=data.shape

T=[20]
for t in T:
    filtered[:,:,:,t]=low_pass_with_cutoff(filtered[:,:,:,t],120,124)

show_img(filtered[:,:,20,20],additional_data=data[:,:,20,20],name='filter.png')
