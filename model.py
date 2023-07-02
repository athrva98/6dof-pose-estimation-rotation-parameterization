# defining the backbone neural network and the rotation head
import numpy as np
from torchinfo import summary
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import os
from torch.autograd import Variable

use_cuda = torch.cuda.is_available()

# Code adapted from "https://github.com/LeeJunHyun/Image_Segmentation"

os.system('clear')

def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)

class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )


    def forward(self,x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out, scale_factor=2):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x


        



class single_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(single_conv,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.conv(x)
        return x

class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi

class rotationHead(nn.Module):
  def __init__(self, _type: int = 0, lineMod : bool = False):
    super(rotationHead,self).__init__()
    if lineMod:
        in_features = 20
    else:
        in_features = 16
    if _type == 0:
      '''
      We assume that the R \in R^{3x3} and t \in R^{3}
      '''
      self.rotation_parameterization = 'RotMat12'
      self.flatten = nn.Flatten()
      self.head = nn.Linear(in_features=in_features, out_features=12)
    if _type == 1:
      '''
      We assume that the Neural net optimization is on the tangent space of SO3
      Then, angular velocity w = (wx, wy, wz) and t \in R^{3}
      '''
      self.rotation_parameterization = 'AngVel6'
      self.flatten = nn.Flatten()
      self.head = nn.Linear(in_features=in_features, out_features=6)
    if _type == 2:
      '''
      We assume that the Neural net optimization is on the tangent space of S3
      and the network learns the exponential map to give us the angle axis version
      u = (xi, yj, zk), theta \in (0, 2*pi) and t \in R^{3}
      '''
      self.roation_parameterization = 'quat7'
      self.flatten = nn.Flatten()
      self.head = nn.Linear(in_features=in_features, out_features=7)
    
    if _type == 3:
        '''
        We consider a generic Affine transformation with no constraints on being in SE(3)
        '''
        self.rotation_parameterization = 'Affine16'
        self.flatten = nn.Flatten()
        self.head = nn.Linear(in_features=in_features, out_features=16)
    
  def forward(self, x):
    x = self.flatten(x)
    x = self.head(x)
    return x

class NeuralNet(nn.Module):
    def __init__(self,in_channels=3,out_channels=1):
        super(NeuralNet,self).__init__()
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=in_channels,ch_out=64)
        self.Conv2 = conv_block(ch_in=64,ch_out=128)
        self.Conv3 = conv_block(ch_in=128,ch_out=256)
        self.Conv4 = conv_block(ch_in=256,ch_out=256)
        self.Conv5 = conv_block(ch_in=256,ch_out=256)
        self.Conv6 = conv_block(ch_in=384,ch_out=32)
        self.pose_conv1 = nn.Conv2d(in_channels=256, out_channels=1,
                                        kernel_size=1,stride=2,
                                        padding=0,bias=True)
        self.pose_conv2 = nn.Conv2d(in_channels=1, out_channels=1,
                                        kernel_size=1,stride=2,
                                        padding=0,bias=True)
        self.pose_conv3 = nn.Conv2d(in_channels=1, out_channels=1,
                                        kernel_size=1,stride=2,
                                        padding=0,bias=True)

        self.Up5 = up_conv(ch_in=256,ch_out=128)
        self.Up6 = up_conv(ch_in=32,ch_out=out_channels, scale_factor=8)
        self.Att5 = Attention_block(F_g=128,F_l=256,F_int=64)


    def forward(self,x):
        # encoding path
        x1 = self.Conv1(x)
        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)
        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)
        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5,x=x4)
        pose = self.pose_conv1(x4)
        pose = self.pose_conv2(pose)
        pose = self.pose_conv3(pose)
        d5 = torch.cat((x4,d5),dim=1) 
        d5 = self.Conv6(d5)
        mask = self.Up6(d5)
        return mask, pose

if __name__ == '__main__':
  backbone = NeuralNet(in_channels=6, out_channels=1)
  dummy_input = torch.from_numpy(np.random.randint(low=0, high= 255, size=(4, 3, 256, 256)))
  print(summary(backbone, input_size=(1, 6, 256, 256)))