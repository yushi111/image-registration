from utils import *

data=get_data()
data_enhanced=linear_mapping(data)
print(data_enhanced[:,98,20,15])
show_img(data=data[:,:,20,15],additional_data=data_enhanced[:,:,20,15])