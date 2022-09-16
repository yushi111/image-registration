import numpy as np
import mat73

def get_ref(img4d):
    """
    requirement: input is a 4d image 
    output: find average at the time dimension
    """
    return np.average(img4d,axis=3)

def get_data(path=None):
    if not path:
        path='/ll3/data/3/human/20220426_GIMRI_fMRI/preprocessed/s019a1001.mat'
    data=mat73.loadmat(path)
    return data