import torch
import torch.nn as nn
import torch.nn.functional as functional

def check_resize(data):
    B,C,N,W,D=data.shape
    if N%2==1:
        data=torch.cat((data,torch.zeros(B,C,1,W,D,dtype=data.dtype,device=data.device)),dim=2)
        B,C,N,W,D=data.shape
    if W%2==1:
        data=torch.cat((data,torch.zeros(B,C,N,1,D,dtype=data.dtype,device=data.device)),dim=3)
        B,C,N,W,D=data.shape
    if D%2==1:
        data=torch.cat((data,torch.zeros(B,C,N,W,1,dtype=data.dtype,device=data.device)),dim=4)
        B,C,N,W,D=data.shape
    return data

def add_conv_stage(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=True, useBN=False):
  if useBN:
    return nn.Sequential(
      nn.Conv3d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias,dtype=torch.double),
      nn.BatchNorm3d(dim_out),
      nn.LeakyReLU(0.1),
      nn.Conv3d(dim_out, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias,dtype=torch.double),
      nn.BatchNorm3d(dim_out),
      nn.LeakyReLU(0.1)
    )
  else:
    return nn.Sequential(
      nn.Conv3d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias,dtype=torch.double),
      nn.ReLU(),
      nn.Conv3d(dim_out, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias,dtype=torch.double),
      nn.ReLU()
    )


def upsample(ch_coarse, ch_fine):
  return nn.Sequential(
    nn.ConvTranspose3d(ch_coarse, ch_fine, 4, 2, 1, bias=False,dtype=torch.double),
    nn.ReLU()
  )

class Net(nn.Module):
  def __init__(self, K=5,useBN=False):
    super(Net, self).__init__()

    self.conv1   = add_conv_stage(1, 32, useBN=useBN)
    self.conv2   = add_conv_stage(32, 64, useBN=useBN)
    self.conv3   = add_conv_stage(64, 128, useBN=useBN)
    self.conv4   = add_conv_stage(128, 256, useBN=useBN)
    self.conv5   = add_conv_stage(256, 512, useBN=useBN)

    self.conv4m = add_conv_stage(512, 256, useBN=useBN)
    self.conv3m = add_conv_stage(256, 128, useBN=useBN)
    self.conv2m = add_conv_stage(128,  64, useBN=useBN)
    self.conv1m = add_conv_stage( 64,  32, useBN=useBN)

    self.conv0  = nn.Sequential(
        nn.Conv3d(32, K, 3, 1, 1,dtype=torch.double),
        nn.Sigmoid(),
        #nn.Softmax3d()
    )
    self.pad = nn.ConstantPad3d(3,0)
    self.max_pool = nn.MaxPool3d(2)

    self.upsample54 = upsample(512, 256)
    self.upsample43 = upsample(256, 128)
    self.upsample32 = upsample(128,  64)
    self.upsample21 = upsample(64 ,  32)
    ## weight initialization
    for m in self.modules():
      if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
        if m.bias is not None:
          m.bias.data.zero_().to(torch.double)
        nn.init.kaiming_normal_(m.weight)
        m.weight.to(torch.double)
    #self.Kconst = torch.tensor(config.K).float()
    #self.cropped_seg = torch.zeros(config.BatchSize,config.K,config.inputsize[0],config.inputsize[1],(config.radius-1)*2+1,(config.radius-1)*2+1)
    #self.loss = NCutsLoss()


  def forward(self, x):#, weight):
    #sw = weight.sum(-1).sum(-1)
    conv1_out = self.conv1(x)
    #return self.upsample21(conv1_out)
    conv2_out = self.conv2(self.max_pool(conv1_out))
    conv3_out = self.conv3(self.max_pool(conv2_out))
    conv4_out = self.conv4(self.max_pool(conv3_out))
    conv5_out = self.conv5(self.max_pool(conv4_out))
    #print(conv1_out.shape)
    #print(conv2_out.shape)
    # print(conv3_out.shape)
    # print(conv5_out.shape)
    # print(self.upsample54(conv5_out).shape)
    conv5m_out = torch.cat((self.upsample54(conv5_out), conv4_out), 1)
    conv4m_out = self.conv4m(conv5m_out)
    # print(self.upsample43(conv4m_out).shape)
    conv4m_out_ = torch.cat((self.upsample43(conv4m_out), conv3_out), 1)
    conv3m_out = self.conv3m(conv4m_out_)

    conv3m_out_ = torch.cat((self.upsample32(conv3m_out), conv2_out), 1)
    conv2m_out = self.conv2m(conv3m_out_)

    conv2m_out_ = torch.cat((self.upsample21(conv2m_out), conv1_out), 1)
    conv1m_out = self.conv1m(conv2m_out_)

    conv0_out = self.conv0(conv1m_out)
    #conv0_softmax=nn.functional.softmax(conv0_out,1)
    conv0_softmax=nn.functional.gumbel_softmax(conv0_out,tau=0.01,hard=True,dim=1)
    padded_seg = self.pad(conv0_out)
    '''for m in torch.arange((config.radius-1)*2+1,dtype=torch.long):
        for n in torch.arange((config.radius-1)*2+1,dtype=torch.long):
            self.cropped_seg[:,:,:,:,m,n]=padded_seg[:,:,m:m+conv0_out.size()[2],n:n+conv0_out.size()[3]].clone()
    multi1 = self.cropped_seg.mul(weight)
    multi2 = multi1.view(multi1.shape[0],multi1.shape[1],multi1.shape[2],multi1.shape[3],-1).sum(-1).mul(conv0_out)
    multi3 = sum_weight.mul(conv0_out)
    assocA = multi2.view(multi2.shape[0],multi2.shape[1],-1).sum(-1)
    assocV = multi3.view(multi3.shape[0],multi3.shape[1],-1).sum(-1)
    assoc = assocA.div(assocV).sum(-1)
    loss = self.Kconst - assoc'''
    #loss = self.loss(conv0_out, padded_seg, weight, sw)
    return [conv0_softmax,nn.functional.softmax(conv0_out,1)]