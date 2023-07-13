import os
import torch

from Networks.UNet import UNetRes as net
from Networks.Utils import Utils_Model
from FDL.FDL_Model import FDLModel
import Networks.Basic_Block as B

'''
---------------------------------------------------------------------------------------------------------------------------
CLASS JointOptimizationFDL
    A Pytorch module to unrolled the ADMM FDL optimization algorithm with a deep synthetiser for light field reconstruction
---------------------------------------------------------------------------------------------------------------------------
'''
class JointOptimizationFDLShiftCoordinatesPretrained(torch.nn.Module):
    def __init__(self, nb_channels=3, nb_iteration=12, nb_fdl=30, rho_initial_val=1.0, device='cpu'):
        super(JointOptimizationFDLShiftCoordinatesPretrained,self).__init__()
        self.nb_iteration=nb_iteration
        self.device = device
        self.nb_fdl = nb_fdl
        self.nb_channels = nb_channels
        
        self.d_fdl = torch.arange(-2, 1, 3.0/nb_fdl)
        
        self.denoiser = net(in_nc=nb_channels*nb_fdl+1, out_nc=nb_channels*nb_fdl, nc=[128, 128, 256, 512], nb=4, act_mode='R', downsample_mode="strideconv", upsample_mode="convtranspose")
        self.synthetiser = net(in_nc=nb_channels*nb_fdl+1, out_nc=nb_channels*nb_fdl, nc=[128, 128, 256, 512], nb=4, act_mode='R', downsample_mode="strideconv", upsample_mode="convtranspose")
        self.fdl_model = FDLModel(self.d_fdl, device=device)

        self.rhos = torch.nn.ParameterList()
        for iteration in range(nb_iteration+1):
            self.rhos += [torch.nn.Parameter(data=torch.tensor(rho_initial_val), requires_grad=True)]   
       
    def add_coordinates_layer(self):
        pretrained_weights = self.synthetiser.m_head.weight
        self.synthetiser.m_head = B.conv(self.nb_channels*self.nb_fdl+3, 128, bias=False, mode='C').to(self.device)
        self.synthetiser.m_head.weight.data.zero_()
        self.synthetiser.m_head.weight.data[:,:-3] = pretrained_weights.data[:,:-1]
        self.synthetiser.m_head.weight.data[:,-1:] = pretrained_weights.data[:,-1:]

    ''' Generate views from a set of FDL '''
    def generate_views(self, fdl, render_params):
        return self.fdl_model.fdl_synthetiser_shift_coordinates_views(fdl, render_params, self.synthetiser)

    ''' Execute the entire unrolled ADMM FDL optimization algorithm '''
    def forward(self, fdl, inputs, views_params):
        v = fdl
        u = 0
        y = torch.fft.fftn(inputs, dim=(-2, -1))
        for iteration in range(self.nb_iteration):
            x = self.fdl_model.fdl_proximal_operator(v - u / self.rhos[iteration], y, self.rhos[iteration], views_params)
            x_denoise = x + u / self.rhos[iteration]
            x_reshaped = x_denoise.view(*x_denoise.shape[0:1],-1,*x_denoise.shape[-2:])
            v = Utils_Model.test_mode(self.denoiser, x_reshaped, mode=2, refield=32, min_size=1024, modulo=16, iteration=iteration)
            v = v.view_as(x)
            u = u + self.rhos[iteration] * (x - v) 
        return v