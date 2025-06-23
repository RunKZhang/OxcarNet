import sys

import torch
import torch.nn as nn
from torch.nn import functional as F
from torchinfo import summary



from oxcar_DL.models.SincConv2D import SincConv2D



class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm: int = 1, **kwargs):
        self.max_norm = max_norm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return super(Conv2dWithConstraint, self).forward(x)

class Spatial_Attn(nn.Module):
    def __init__(self, in_dim, num_electrodes, signal_length):
        super(Spatial_Attn, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=(num_electrodes, 1))
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=(num_electrodes,1))
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=(num_electrodes,1))
        # self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

        self.recover = nn.Upsample(size=[num_electrodes, signal_length], mode='bilinear')
    
    def forward(self, x):
        B, C, H, W = x.size()
        query = self.query_conv(x).squeeze().permute(0,2,1)
        # query = self.query_conv(x)
        key = self.key_conv(x).squeeze()
        qu_ke_map = torch.bmm(query, key)
        attention = self.softmax(qu_ke_map)
        value = self.value_conv(x).squeeze()
        out = torch.bmm(value, attention.permute(0,2,1))
        out = out.view(B,C,1,W)
        out = self.recover(out)
        out = out + x
        return out

class Temp_Attn(nn.Module):
    def __init__(self, in_dim, len_patch: int, stride: int, signal_length: int):
        super(Temp_Attn, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=len_patch, stride=stride)
        self.key_conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=len_patch, stride=stride)
        self.value_conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=len_patch, stride=stride)
        # self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

        self.recover = nn.Upsample(size=signal_length)
    def forward(self, x):
        B, C, T = x.size()
        query = self.query_conv(x).permute(0, 2, 1)
        key = self.key_conv(x)
        qu_ke_map = torch.bmm(query,key)
        attention = self.softmax(qu_ke_map)
        value = self.value_conv(x)
        out = torch.bmm(value, attention.permute(0,2,1))
        out = self.recover(out)
        out = out + x
        return out

class Chn_Attn(nn.Module):
    def __init__(self, in_dim):
        super(Chn_Attn, self).__init__()
        self.in_dim = in_dim

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, C, T = x.size()
        query = x
        key = x
        value = x
        qu_ke_map = torch.bmm(key, query.permute(0, 2, 1))
        attention = self.softmax(qu_ke_map)
        out = torch.bmm(attention, value)
        out = out + x
        return out

class new_TEMP_Module(nn.Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(new_TEMP_Module, self).__init__()
        self.chanel_in = in_dim
        self.num_electrodes = 19

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        # self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=(self.num_electrodes, 1))
        # self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=(self.num_electrodes, 1))
        # self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=(self.num_electrodes, 1))
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x):
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out

class new_CHN_Module(nn.Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(new_CHN_Module, self).__init__()
        self.chanel_in = in_dim


        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)
    def forward(self,x):
        # original
        # m_batchsize, C, height, width = x.size()
        # changed
        m_batchsize, C, length = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, length)

        out = self.gamma*out + x
        return out    
            
class OxcarNet(nn.Module):
    def __init__(self, num_classes: int, 
                 num_electrodes: int, 
                 dilated_factor: int, 
                 sinc_length_factor: int,
                 chunk_size: int = 256,
                 dropout: float = 0.25
                 ):        
        super().__init__()

        self.F1 = dilated_factor
        self.chunk_size = chunk_size
        self.num_electrodes = num_electrodes
        self.dropout = dropout

        self.sinc_block = nn.Sequential(
            SincConv2D(in_channels=1, 
                       out_channels=dilated_factor, 
                       kernel_size= (chunk_size // sinc_length_factor + 1), 
                       stride=1,
                       padding='same',
                       fs=256),
        )

        self.spat_block = nn.Sequential(
            # Spatial_Attn(self.F1, num_electrodes=self.num_electrodes, signal_length=self.chunk_size),
            # PAM_Module(dilated_factor),
            # CAM_Module(dilated_factor),
            Conv2dWithConstraint(self.F1,
                                 self.F1, (self.num_electrodes, 1),
                                 max_norm=1,
                                 stride=1,
                                 padding=(0, 0),
                                 groups=self.F1,
                                 bias=False), 
            nn.BatchNorm2d(self.F1, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(), 
            nn.AvgPool2d((1, 4), stride=4), 
            # nn.Dropout(p=dropout)
        )

        self.temp_block = nn.Sequential(
            # Temp_Attn(dilated_factor, len_patch=4, stride=4, signal_length = self.chunk_size // 4),
            new_TEMP_Module(dilated_factor),
            # nn.Conv1d(self.F1, self.F1, 4, stride=4, padding=(0, self.kernel_1 // 2), bias=False),
            # nn.Conv1d(self.F1, self.F1, 4, stride=4, bias=False),
            nn.Conv2d(self.F1, self.F1, (1, 4), 4, bias=False),
            # nn.BatchNorm1d(self.F1, momentum=0.01, affine=True, eps=1e-3),
            nn.BatchNorm2d(self.F1, momentum=0.01, affine=True, eps=1e-3),
            # nn.LayerNorm([self.F1, self.chunk_size // 16]),
            nn.ELU(),

        )
        
        self.chn_block = nn.Sequential(
            # Chn_Attn(dilated_factor),
            new_CHN_Module(dilated_factor),
            nn.Conv1d(self.F1, self.F1, kernel_size=16, stride=1, bias=False),
            nn.Dropout(p=dropout)
        )
        
    def forward(self, x):
        B, C, H, W = x.size()
        x = self.sinc_block(x)
        
        x = self.spat_block(x)
        # x = x.squeeze()
        x = self.temp_block(x)
        x = x.squeeze()
        x = self.chn_block(x)
        x = x.view(B, -1)
        embedding = x        
        return embedding
    

if __name__ == "__main__":
    model = nn.Sequential(
        OxcarNet(num_classes=2, num_electrodes=19, dilated_factor=32, chunk_size=256, sinc_length_factor=4),
        # classifier(512, 2)
    )
    # model = Spatial_Attn(32, num_electrodes=19, signal_length=256)
    # model = Temp_Attn(32, 4, 4, 64)
    # model = EEGNet(chunk_size=256, num_electrodes=19)
    x = torch.randn([32, 1, 19, 256], requires_grad=False)    
    # x = torch.randn([32, 32, 64], requires_grad=False)    
    # y = model(x)
    # print(y.shape)
    summary(model, input_size=(32, 1, 19, 256))