from concurrent.futures import process
import imp
import numpy as np
import mat73
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import torch
from torch import F

def get_ref(img4d):
    """
    requirement: input is a 4d image 
    output: find average at the time dimension
    """
    return np.average(img4d,axis=3)

def get_data(path=None):
    """
    path: path to the data mat file
    output: numpy 4d array
    caveats: mat files may have different key name
    """
    if not path:
        path='/ll3/data/3/human/20220426_GIMRI_fMRI/preprocessed/s019a1001.mat'
    data=mat73.loadmat(path)
    return data['mat']

def show_img(data,name="test.jpg",additional_data=None,argv=None):
    """
    data: if data is a 2d image, directly show it, 
          if it's a 3d image, argv is a list that specipy the third dimension to be plotted (size(argv<=4))
    name: name of output file
    additional_data: can add an additional 2d image if data is 2d (for comparison)
    argv: list of optional slices
    output: an image file
    """
    if len(data.shape)==2:
        if additional_data is not None:
            fig,ax=plt.subplots(1,2)
            plt.sca(ax[0])
            plt.imshow(data,cmap='gray')
            plt.sca(ax[1])
            plt.imshow(additional_data,cmap='gray')
            plt.savefig(name)
        else:
            fig,ax=plt.subplots()
            plt.imshow(data,cmap='gray')
            plt.savefig(name)
    elif len(data.shape)==3:
        fig,ax=plt.subplots(2,2)
        for i,slice in enumerate(argv):
            plt.sca(ax[i//2][i%2])
            plt.imshow(data[:,:,slice],cmap='gray')

        plt.savefig(name)
    else:
        raise NotImplementedError("Only support 2/3 D data")

def low_pass_3d(data,sigma):
    """
    data: 3d image data
    sigma: sigmas for gaussian filter
    output: data after low pass filter
    """
    filtered_data=gaussian_filter(data,sigma=sigma)
    return filtered_data

def sampler(data,displacement):
    """
    data: 3d image (N,W,D) tensor
    displacement: displacement vector for each dimensions (N,W,D,3) tensor
    """
    data=torch.unsqueeze(data,0)
    data=torch.unsqueeze(data,0)
    displacement=torch.unsqueeze(displacement,0)
    #normalize the displacement
    
    out=F.grid_sample(data,displacement)