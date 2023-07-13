import math
import numpy as np
import torch

from Networks.Utils import Utils_Model
from Utils.Fourier_Utils import gen_frequency_grid

'''
---------------------
CLASS FDLModel
    FDL pytorch model
---------------------
'''
class FDLModel(torch.nn.Module):
    def __init__(self, d_fdl=30, device='cpu'):
        super(FDLModel, self).__init__()
        self.device = device
        self.d_fdl = d_fdl
        self.freq_blk_sz = 4096
        self.register_buffer('w_x', None, persistent=False)
        self.register_buffer('w_y', None, persistent=False)

    ''' Compute the proximal operator of a FDL data-fit term '''     
    def fdl_proximal_operator(self, x, y, rho, views_params):
        x = torch.fft.fftn(x, dim=(-2, -1))

        n_batch = y.shape[0]
        n_chan = y.shape[1]
        n_input_views = np.prod(y.shape[2:-2])
        n_freq_x = y.shape[-1]
        n_freq_y = y.shape[-2]
        n_disp = len(self.d_fdl)

        x_c = math.ceil((n_freq_x + 1) / 2)
        y_c = math.ceil((n_freq_y + 1) / 2)
        even_fft = [1 - i % 2 for i in y.shape[-2:]]
        n_freq_proc = n_freq_y * (x_c - 1) + y_c 

        self.w_x = self.w_x.transpose(0, 1).reshape(-1, 1, 1)
        self.w_y = self.w_y.transpose(0, 1).reshape(-1, 1, 1)

        views_params.to(self.device)
        views_params.set_fdl_disparities(self.d_fdl)

        y = y.transpose(-1, -2)
        y = y.roll(shifts=(x_c - 1, y_c - 1), dims=(-2, -1))
        y = y.reshape(n_batch, n_chan, n_input_views, n_freq_x * n_freq_y, 1).transpose(2, 3)
        y = y[..., :n_freq_proc, :, :]

        x = x.transpose(-1, -2)
        x = x.roll(shifts=(x_c - 1, y_c - 1), dims=(-2, -1))
        x = x.reshape(n_batch, n_chan, n_disp, n_freq_x * n_freq_y, 1).transpose(2, 3)
        x = rho * x[..., :n_freq_proc, :, :].to(self.device)

        lambda_x_fdl_source_freqs = 0.0
        fdl = torch.empty(n_batch, n_chan, n_freq_proc, n_disp, device=self.device, dtype=torch.complex64)
        Reg = rho * torch.eye(n_disp, device=self.device, dtype=torch.complex64).view(1, 1, 1, n_disp, n_disp)

        for freq_st in range(0, n_freq_proc, self.freq_blk_sz):
            freq_end = min(freq_st + self.freq_blk_sz, n_freq_proc)

            A = views_params.make_fdl_data_matrix(self.w_x[freq_st:freq_end, :, :], self.w_y[freq_st:freq_end, :, :])

            AT = A.transpose(-1, -2).conj()
            ATA = AT.matmul(A)
            ATb = AT.matmul(y[..., freq_st:freq_end, :, :])
            del A, AT

            lambda_x_fdl_source_freqs = x[..., freq_st:freq_end, :, :]
            
            fdl[..., freq_st: freq_end, :] = ((ATA + Reg).inverse().matmul(ATb + lambda_x_fdl_source_freqs))[..., 0]

        del ATA, ATb

        fdl = torch.cat([fdl, torch.zeros(n_batch, n_chan, n_freq_x * n_freq_y - n_freq_proc, n_disp,
                                          dtype=torch.complex64, device=self.device)], -2)
        fdl = fdl.view(n_batch, n_chan, n_freq_x, n_freq_y, n_disp)
        fdl = fdl.permute(0, 1, 4, 3, 2)
        fdl[..., y_c:, x_c-1:] = fdl[..., even_fft[0]:y_c-1, even_fft[1]:x_c].flip(-1, -2).conj()
        fdl[..., even_fft[0]:y_c, x_c:] = fdl[..., y_c-1:, even_fft[1]:x_c-1].flip(-1, -2).conj()
        fdl = fdl.roll(shifts=(1-y_c, 1-x_c), dims=(-2, -1))

        x = torch.fft.ifftn(fdl, dim=(-2, -1)).real
        return x
    
    ''' Compute views from a set of FDL '''
    def fdl_render(self, x, views_params):
        x = torch.fft.fftn(x, dim=(-2, -1))

        n_batch = x.shape[0]
        n_chan = x.shape[1]
        n_disp = x.shape[2]
        n_freq_x = x.shape[-1]
        n_freq_y = x.shape[-2]
    
        n_output_views = views_params.num_views()
    
        x_c = math.ceil((n_freq_x + 1) / 2)
        y_c = math.ceil((n_freq_y + 1) / 2)
        even_fft = [1 - i % 2 for i in x.shape[-2:]]
        n_freq_proc = n_freq_y * (x_c - 1) + y_c

        self.w_x = self.w_x.transpose(0, 1).reshape(-1, 1, 1)
        self.w_y = self.w_y.transpose(0, 1).reshape(-1, 1, 1)

        views_params.to(self.device)
        views_params.set_fdl_disparities(self.d_fdl)

        x = x.transpose(-1, -2)
        x = x.roll(shifts=(x_c - 1, y_c - 1), dims=(-2, -1))
        x = x.reshape(n_batch, n_chan, n_disp, n_freq_x * n_freq_y, 1).transpose(2, 3)
        x = x[..., :n_freq_proc, :, :].to(self.device)

        views_fft = torch.empty(n_batch, n_chan, n_freq_proc, n_output_views, device=self.device, dtype=torch.complex64)

        for freq_st in range(0, n_freq_proc, self.freq_blk_sz):
            freq_end = min(freq_st + self.freq_blk_sz, n_freq_proc)

            A = views_params.make_fdl_data_matrix(self.w_x[freq_st:freq_end, :, :], self.w_y[freq_st:freq_end, :, :])
            views_fft[..., freq_st: freq_end, :] = A.matmul(x[..., freq_st:freq_end, :, :])[..., 0]

        views_fft = torch.cat([views_fft, torch.zeros(n_batch, n_chan, n_freq_x * n_freq_y - n_freq_proc, n_output_views,
                                                      dtype=torch.complex64, device=self.device)], -2)
        views_fft = views_fft.view(n_batch, n_chan, n_freq_x, n_freq_y, n_output_views)
        views_fft = views_fft.permute(0, 1, 4, 3, 2)
        views_fft[..., y_c:, x_c - 1:] = views_fft[..., even_fft[0]:y_c - 1, even_fft[1]:x_c].flip(-1, -2).conj()
        views_fft[..., even_fft[0]:y_c, x_c:] = views_fft[..., y_c - 1:, even_fft[1]:x_c - 1].flip(-1, -2).conj()
        views_fft = views_fft.roll(shifts=(1 - y_c, 1 - x_c),dims=(-2, -1))
        
        return views_fft
        
    ''' Synthesize views from a set of FDL '''
    def fdl_synthetiser_shift_views(self, x, views_params, synthetiser):
        x = torch.fft.fftn(x, dim=(-2, -1))

        n_batch = x.shape[0]
        n_chan = x.shape[1]
        n_disp = x.shape[2]
        n_freq_x = x.shape[-1]
        n_freq_y = x.shape[-2]
    
        n_output_views = views_params.num_views()
    
        x_c = math.ceil((n_freq_x + 1) / 2)
        y_c = math.ceil((n_freq_y + 1) / 2)
        even_fft = [1 - i % 2 for i in x.shape[-2:]]
        n_freq_proc = n_freq_y * n_freq_x
        
        self.w_x, self.w_y = gen_frequency_grid(n_freq_x, n_freq_y, half_x_dim=False, centered_zero=True, device=self.device)

        self.w_x = self.w_x.transpose(0, 1).reshape(-1, 1, 1)
        self.w_y = self.w_y.transpose(0, 1).reshape(-1, 1, 1)

        views_params.to(self.device)
        views_params.set_fdl_disparities(self.d_fdl)

        x = x.transpose(-1, -2)
        x = x.roll(shifts=(x_c - 1, y_c - 1), dims=(-2, -1))
        x = x.reshape(n_batch, n_chan, n_disp, n_freq_x * n_freq_y).transpose(2, 3).unsqueeze(-2).expand(n_batch, n_chan, n_freq_x * n_freq_y, n_output_views, n_disp)
        x = x[..., :n_freq_proc, :, :].to(self.device)

        views_fft = torch.empty(n_batch, n_chan, n_freq_y, n_freq_x, n_output_views, device=self.device, dtype=torch.complex64)
        
        shifted_fdl = torch.zeros_like(x).to(self.device)

        for freq_st in range(0, n_freq_proc, self.freq_blk_sz):
            freq_end = min(freq_st + self.freq_blk_sz, n_freq_proc)

            A = views_params.make_fdl_data_matrix(self.w_x[freq_st:freq_end, :, :], self.w_y[freq_st:freq_end, :, :])
            shifted_fdl[..., freq_st:freq_end, :, :] = A*x[..., freq_st:freq_end, :, :]

        shifted_fdl = shifted_fdl.view(n_batch, n_chan, n_freq_x, n_freq_y, n_output_views, n_disp)
        shifted_fdl = shifted_fdl.permute(0, 1, 4, 5, 3, 2) 
        shifted_fdl[..., y_c:, x_c - 1:] = shifted_fdl[..., even_fft[0]:y_c - 1, even_fft[1]:x_c].flip(-1, -2).conj()
        shifted_fdl[..., even_fft[0]:y_c, x_c:] = shifted_fdl[..., y_c - 1:, even_fft[1]:x_c - 1].flip(-1, -2).conj()
        shifted_fdl = shifted_fdl.roll(shifts=(1 - y_c, 1 - x_c),dims=(-2, -1))

        for n in range(shifted_fdl.shape[2]):
            ifft_shifted_fdl = shifted_fdl[:,:,n]
            ifft_shifted_fdl = torch.fft.ifftn(ifft_shifted_fdl, dim=(-2, -1)).real
            ifft_shifted_fdl_reshaped = ifft_shifted_fdl.view(*ifft_shifted_fdl.shape[0:1],-1,*ifft_shifted_fdl.shape[-2:])                
            ifft_synthetised_fdl = Utils_Model.test_mode(synthetiser, ifft_shifted_fdl_reshaped, mode=2, refield=32, min_size=1024, modulo=16, iteration=0)
            ifft_synthetised_fdl = ifft_synthetised_fdl.view_as(ifft_shifted_fdl)
            fft_shifted_fdl = torch.fft.fftn(ifft_synthetised_fdl, dim=(-2, -1))
            views_fft[:,:,:,:,n] = torch.sum(fft_shifted_fdl,dim=2)
        
        views_fft = views_fft.permute(0,1,4,2,3).contiguous()
        
        return views_fft

    ''' Synthesize views from a set of FDL '''
    def fdl_synthetiser_shift_coordinates_views(self, x, views_params, synthetiser): 
        x = torch.fft.fftn(x, dim=(-2, -1))

        n_batch = x.shape[0]
        n_chan = x.shape[1]
        n_disp = x.shape[2]
        n_freq_x = x.shape[-1]
        n_freq_y = x.shape[-2]
    
        n_output_views = views_params.num_views()
    
        x_c = math.ceil((n_freq_x + 1) / 2)
        y_c = math.ceil((n_freq_y + 1) / 2)
        even_fft = [1 - i % 2 for i in x.shape[-2:]]
        n_freq_proc = n_freq_y * n_freq_x
        
        self.w_x, self.w_y = gen_frequency_grid(n_freq_x, n_freq_y, half_x_dim=False, centered_zero=True, device=self.device)

        self.w_x = self.w_x.transpose(0, 1).reshape(-1, 1, 1)
        self.w_y = self.w_y.transpose(0, 1).reshape(-1, 1, 1)

        views_params.to(self.device)
        views_params.set_fdl_disparities(self.d_fdl)

        x = x.transpose(-1, -2)
        x = x.roll(shifts=(x_c - 1, y_c - 1), dims=(-2, -1))
        x = x.reshape(n_batch, n_chan, n_disp, n_freq_x * n_freq_y).transpose(2, 3).unsqueeze(-2).expand(n_batch, n_chan, n_freq_x * n_freq_y, n_output_views, n_disp)
        x = x[..., :n_freq_proc, :, :].to(self.device)

        views_fft = torch.empty(n_batch, n_chan, n_freq_y, n_freq_x, n_output_views, device=self.device, dtype=torch.complex64)

        for iter_views in range(0,x.shape[-2]):
            shifted_fdl = torch.zeros_like(x[...,iter_views,:]).to(self.device)
            for freq_st in range(0, n_freq_proc, self.freq_blk_sz):
                freq_end = min(freq_st + self.freq_blk_sz, n_freq_proc)
                A = views_params.make_fdl_data_matrix(self.w_x[freq_st:freq_end, :, :], self.w_y[freq_st:freq_end, :, :])
                shifted_fdl[...,freq_st:freq_end, :] = A[:,iter_views]*x[..., freq_st:freq_end, iter_views, :]
                
            shifted_fdl = shifted_fdl.view(n_batch, n_chan, n_freq_x, n_freq_y, n_disp)
            shifted_fdl = shifted_fdl.permute(0, 1, 4, 3, 2) 
            shifted_fdl[..., y_c:, x_c - 1:] = shifted_fdl[..., even_fft[0]:y_c - 1, even_fft[1]:x_c].flip(-1, -2).conj()
            shifted_fdl[..., even_fft[0]:y_c, x_c:] = shifted_fdl[..., y_c - 1:, even_fft[1]:x_c - 1].flip(-1, -2).conj()
            shifted_fdl = shifted_fdl.roll(shifts=(1 - y_c, 1 - x_c),dims=(-2, -1))

            current_u = views_params.u[iter_views] / float(math.sqrt(int(views_params.u.shape[0]/2.0)))
            current_v = views_params.v[iter_views] / float(math.sqrt(int(views_params.v.shape[0]/2.0)))
            ifft_shifted_fdl = shifted_fdl
            ifft_shifted_fdl = torch.fft.ifftn(ifft_shifted_fdl, dim=(-2, -1)).real
            ifft_shifted_fdl_reshaped = ifft_shifted_fdl.view(*ifft_shifted_fdl.shape[0:1],-1,*ifft_shifted_fdl.shape[-2:])
            channel_u = torch.full((ifft_shifted_fdl_reshaped.shape[0],1,*ifft_shifted_fdl_reshaped.shape[2:]),current_u.item()).to(self.device)
            channel_v = torch.full((ifft_shifted_fdl_reshaped.shape[0],1,*ifft_shifted_fdl_reshaped.shape[2:]),current_v.item()).to(self.device)
            inp_net = torch.cat([ifft_shifted_fdl_reshaped,channel_u,channel_v],dim=1)
            ifft_synthetised_fdl = Utils_Model.test_mode(synthetiser, inp_net, mode=2, refield=32, min_size=1024, modulo=16, iteration=0)
            ifft_synthetised_fdl = ifft_synthetised_fdl.view_as(ifft_shifted_fdl)
            fft_shifted_fdl = torch.fft.fftn(ifft_synthetised_fdl, dim=(-2, -1))
            views_fft[:,:,:,:,iter_views] = torch.sum(fft_shifted_fdl,dim=2)
        views_fft = views_fft.permute(0,1,4,2,3).contiguous()
        
        return views_fft