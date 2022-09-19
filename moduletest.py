from utils import *

data=get_data()
print(data.shape)

filtered=low_pass_3d(data,sigma=3)
show_img(data=data[:,:,20,15],additional_data=filtered[:,:,20,15])

print(data[:,:,20,15],filtered[:,:,20,15])
