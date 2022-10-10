from concurrent.futures import process
import imp
import numpy as np
import mat73
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
import matplotlib.cm as cm
from scipy.ndimage import gaussian_filter
import torch
import torch.nn.functional as F
from math import sqrt,floor,ceil
import nibabel as nib 
from mpl_toolkits.mplot3d import axes3d
from tqdm import tqdm

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
    fig.clear(True)

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

def create_video(img4d,slices=[20],name="slice20.gif"):
    """
    img4d: 4d image with the last dimension as time
    slices: a list which contains the 2d slices to be shown
    name: end with gif
    """
    img = [] # some array of images
    frames = [] # for storing the generated images
    N,S,W,T =img4d.shape
    if len(slices)==1:
        fig, ax = plt.subplots()
        for t in range(T):
            frames.append([ax.imshow(img4d[:,:,slices[0],t],cmap=cm.Greys_r,animated=True)])
    else:
        nrows=floor(sqrt(len(slices)))
        ncols=ceil(sqrt(len(slices)))
        if nrows*ncols<len(slices):
            nrows+=1
        fig, ax = plt.subplots()
        img_data=np.zeros((N*nrows,S*ncols))

        for t in range(T):
            for idx,s in enumerate(slices):
                row_start=idx//ncols
                col_start=idx%ncols
                img_data[row_start*N:(row_start+1)*N,col_start*S:(col_start+1)*S]=img4d[:,:,slices[idx],t]
            frames.append([ax.imshow(img_data,cmap=cm.Greys_r,animated=True)])
            
            
    

    """
    ss=[1,10,11,20]
    for i,f in enumerate(ss):
        for t in range(tt):
            axs.plot(data[:,:,f,t],cmap=cm.Greys_r)
            #frames.append([axs[i].imshow(data[:,:,f,t],cmap=cm.Greys_r,animated=True)])
        #plt.show()
    for s in ss:
        for t in range(tt):
            frames.append([ax.imshow(img4d[:,:,s,t],cmap=cm.Greys_r,animated=True)])
        #plt.show()
    """
    
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

def divergence(field):
    div_value=0
    divs=torch.gradient(field,dim=[0,1,2])
    for div in divs:
        div_value+=torch.sum(torch.abs(div))
    return div_value

def plot_loss(step,loss,epoch,t,div=None):
    fig,ax=plt.subplots()
    ax.plot(step,loss,'r',lw=1)
    if div is not None:
        ax.plot(step,div,'b',lw=1)
    ax.set_title("loss")
    ax.set_xlabel("steps")
    ax.set_ylabel("loss")
    ax.legend(['Total loss','divergence'])
    fig.savefig(f"loss at {epoch}num{t}.png")
    fig.clear(True)

def show_hist(img3d,name='hist.png'):
    """
    img3d: target 3d image to show its voxel intensity distribution
    
    """
    voxels=np.reshape(img3d,(-1,))
    fig,ax=plt.subplots()
    plt.hist(voxels,bins=256)
    plt.xlabel('Intensity')
    plt.ylabel('Count')
    plt.ylim([0,100000])
    plt.savefig(name)

def get_nii_data(path=None):
    if path is None:
        path='file1410.nii.gz'
    nii_img=nib.load(path)
    nii_data=nii_img.get_fdata()
    return nii_data

def plot_3dvector_field(initial_coor,displacement,name='vectorfield'):
    X=15
    Y=15
    Z=10
    fig=plt.figure()
    ax=fig.gca(projection='3d')
    #ax.quiver(initial_coor[::X,::Y,::Z,0],initial_coor[::X,::Y,::Z,1],initial_coor[::X,::Y,::Z,2],displacement[::X,::Y,::Z,0],displacement[::X,::Y,::Z,1],displacement[::X,::Y,::Z,2],length=1.5)
    ax.quiver(initial_coor[160:170,160:170,::Z,0],initial_coor[160:170,160:170,::Z,1],initial_coor[160:170,160:170,::Z,2],displacement[160:170,160:170,::Z,0],displacement[160:170,160:170,::Z,1],displacement[160:170,160:170,::Z,2],length=1,arrow_length_ratio=0.3)
    plt.savefig(name)
    plt.show()


def get_filtering_kernel(N,W,D,start,end):
    """
    generate filtering kernel for blurring
    N,W,D: dimensions of 3d image 
    start: starting position of frequency cutting [0,max(N,W,D)/2]
    end: end position of frequency cutting 
    """
    factors=[N,W,D]
    max_dim=max(factors)
    mask=[np.zeros((N,),dtype=np.float64),np.zeros((W,),dtype=np.float64),np.zeros((D,),dtype=np.float64)]
    
    for i,d in enumerate([N,W,D]):
        factors[i]=d/max_dim
        mask[i][d//2-round(start*factors[i]):d//2+round(start*factors[i])]=1
        slope=1./(round(end*factors[i])-round(start*factors[i]))
        for j in range(round(end*factors[i])-round(start*factors[i])):
            mask[i][d//2-round(end*factors[i])+j]=slope*j
            if d%2==1:
                mask[i][d//2+round(start*factors[i])+j]=1-slope*j
            else:
                mask[i][d//2+round(start*factors[i])+j-1]=1-slope*j
        mask[i]=1-mask[i]
    
    
    xv,yv,zv=np.meshgrid(mask[1],mask[0],mask[2])
    return xv*yv*zv.astype(np.complex64)

def low_pass_with_cutoff(data,start,end):
    """
    data: 3d image data
    sigma: sigmas for gaussian filter
    output: data after low pass filter
    """
    fft_data=np.fft.fftn(data)
    N,W,D=data.shape
    mask=get_filtering_kernel(N,W,D,start,end)
    filtered_data=np.fft.ifftn(mask*fft_data)
    return np.abs(filtered_data)

    
    