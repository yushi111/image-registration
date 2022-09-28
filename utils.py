from concurrent.futures import process
import imp
from pickle import TRUE
import numpy as np
import mat73
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
import matplotlib.cm as cm
from scipy.ndimage import gaussian_filter
import torch
import torch.nn.functional as F

def get_ref(img4d):
    """
    requirement: input is a 4d image 
    output: find average at the time dimension
    """
    return torch.tensor(np.average(img4d,axis=3))

def linear_mapping(img4d):
    """
    linearly increase the intensity range of image
    """
    min_intensity=np.min(img4d)
    max_intensity=np.max(img4d)
    return (img4d-min_intensity)/(max_intensity-min_intensity)*255

def get_data(path=None):
    """
    path: path to the data mat file
    output: numpy 4d array
    caveats: mat files may have different key name
    """
    if not path:
        #path='/ll3/data/3/human/20220426_GIMRI_fMRI/preprocessed/s019a1001.mat'
        path='s019a1001.mat'
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
    
    plt.cla()

def low_pass_3d(data,sigma):
    """
    data: 3d image data
    sigma: sigmas for gaussian filter
    output: data after low pass filter
    """
    filtered_data=gaussian_filter(data,sigma=sigma)
    return filtered_data

def sampler(data,displacement,devices='cpu'):
    """
    data: 3d image (N,W,D) tensor
    displacement: displacement vector for each dimensions (N,W,D,3) tensor
    (N,W,D,0) for z axis, and 1 for y, 2 for x
    """
    N,W,D=data.shape
    data=torch.unsqueeze(data,0)
    data=torch.unsqueeze(data,0)
    
    xaxis=torch.linspace(0,N-1,steps=N,dtype=torch.float64)
    yaxis=torch.linspace(0,W-1,steps=W,dtype=torch.float64)
    zaxis=torch.linspace(0,D-1,steps=D,dtype=torch.float64)
    xx,yy,zz=torch.meshgrid(xaxis,yaxis,zaxis)
    if devices != 'cpu':
        coordinates=torch.stack((zz,yy,xx),dim=3).cuda()
    else:
        coordinates=torch.stack((zz,yy,xx),dim=3)

    total_displacement=coordinates+displacement

    #normalize:
    for i,s in enumerate([D,W,N]):
        total_displacement[:,:,:,i]=2*total_displacement[:,:,:,i]/(s-1)-1
    
    total_displacement=total_displacement.unsqueeze(0)
    out=F.grid_sample(data,total_displacement,padding_mode="border",align_corners=True)
    return out.squeeze()

def create_video(img4d,name="slice20.gif"):
    img = [] # some array of images
    frames = [] # for storing the generated images

    fig, ax = plt.subplots()
    x,y,ss,tt =img4d.shape
    ss=[20]

    """

    ss=[1,10,11,20]
    for i,f in enumerate(ss):
        for t in range(tt):
            axs.plot(data[:,:,f,t],cmap=cm.Greys_r)
            #frames.append([axs[i].imshow(data[:,:,f,t],cmap=cm.Greys_r,animated=True)])
        #plt.show()
    """

    for s in ss:
        for t in range(tt):
            frames.append([ax.imshow(img4d[:,:,s,t],cmap=cm.Greys_r,animated=True)])
        #plt.show()


    ani = animation.ArtistAnimation(fig, frames, interval=250, blit=True,
                                    repeat_delay=100)
    #ani = animation.ArtistAnimation(fig, frames, interval=50, blit=True,
    #                                repeat_delay=1000)
    ani.save(name)

def get_3d_rotation_matrix(angle,axis='x'):
    """
    returns a 3d rotation pytorch tensor, the default axis is x
    """
    R=None
    angle=torch.tensor(angle)*torch.pi/180
    if axis=='z':
        R=torch.tensor([[torch.cos(angle),-torch.sin(angle),0],
                        [torch.sin(angle),torch.cos(angle),0],
                        [0,0,1]])
    elif axis=='y':
        R=torch.tensor([[torch.cos(angle),0,torch.sin(angle)],
                        [0,1,0],
                        [-torch.sin(angle),0,torch.cos(angle)]])
    elif axis=='x':
        R=torch.tensor([[1,0,0],
                        [0,torch.cos(angle),-torch.sin(angle)],
                        [0,torch.sin(angle),torch.cos(angle)]])
    else:
        raise NotImplementedError("axis only support x, y or z")
    
    return R

def get_3d_coordinate(N,W,D,device='cpu'):
    """
    return 3d coordinate tensor, eg: (0,0,0),(0,0,1)...
    """
    xaxis=torch.linspace(0,N-1,steps=N,dtype=torch.float64)
    yaxis=torch.linspace(0,W-1,steps=W,dtype=torch.float64)
    zaxis=torch.linspace(0,D-1,steps=D,dtype=torch.float64)
    xx,yy,zz=torch.meshgrid(xaxis,yaxis,zaxis)
    initial_crd=torch.stack((zz,yy,xx),dim=3).to(device)
    return initial_crd