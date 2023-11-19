import traceback
import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
import copy

from model.decoder import dec_builder
from model.content_encoder import content_enc_builder
from model.references_encoder import comp_enc_builder
from model.Component_Attention_Module import ComponentAttentionModule
from model.memory import Memory
from model.VectorQuantizer import  VectorQuantizer,VectorQuantizerEMA


class Generator(nn.Module):
    """
    Generator
    """

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(Generator,self).__init__()

        # codebook使用VQ-VAE进行编码
        num_embeddings = 524288 # 嵌入向量数量，过多容易过拟合，过少容易欠拟合
        embedding_dim = 1 # 1*1
        commitment_cost = 0.25
        self.vq = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)

        # MHA多头注意力机制，输入input和vq生成的Query
        self.MHA = ComponentAttentionModule()

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True,vq=self.vq,HMA=self.MHA)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input_s,input_c):
        """Standard forward"""
        return self.model(input_s,input_c)

class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False,HMA = None,vq = None):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        self.innermost = innermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]

            self.down_s = nn.Sequential(*[copy.deepcopy(layer) for layer in down])
            self.down_c = nn.Sequential(*[copy.deepcopy(layer) for layer in down])
            self.up = nn.Sequential(*up)
            self.submodule  = submodule
        elif innermost:
             # 原有的下采样层
            down = [downrelu, downconv]
            self.down_s = nn.Sequential(*[copy.deepcopy(layer) for layer in down])
            self.down_c = nn.Sequential(*[copy.deepcopy(layer) for layer in down])          
            # 定义上采样层
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            up = [uprelu, upconv, upnorm]
            self.up = nn.Sequential(*up)
            #生成codebook中最相近的vq
            self.vq = vq
            # MHA多头注意力机制，输入input和vq生成的Query
            self.MHA = HMA
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            self.down_s = nn.Sequential(*[copy.deepcopy(layer) for layer in down])
            self.down_c = nn.Sequential(*[copy.deepcopy(layer) for layer in down])
            self.up = nn.Sequential(*up)
            self.submodule  = submodule

            # if use_dropout:
            #     model = down + [submodule] + up + [nn.Dropout(0.5)]
            # else:
            #     model = down + [submodule] + up

    def forward(self, style , content):
        if self.outermost:
            down_s = self.down_s(style)
            down_c = self.down_c(content)
            submoduled_s,submoduled_c,vq_loss_s,vq_loss_c = self.submodule(down_s , down_c)
            up_s = self.up(submoduled_s)
            up_c = self.up(submoduled_c)
            return up_s,up_c,vq_loss_s,vq_loss_c
        else:   # add skip connections
            if self.innermost:
                # 在最内层先进行下采样
                down_s = self.down_s(style)
                down_c = self.down_c(content)              
                # 然后并行地执行MHA和VQ操作               
                query_s ,vq_loss_s  = self.vq(down_s)
                MHA_s = self.MHA(down_s,query_s)
                query_c ,vq_loss_c  = self.vq(down_c)
                MHA_c = self.MHA(down_c,query_c)
                # 执行上采样
                up_s = self.up(MHA_s)
                up_c = self.up(MHA_c)
                
                return torch.cat([style, up_s], 1) , torch.cat([content , up_c],1),vq_loss_s,vq_loss_c
            else:
                down_s = self.down_s(style)
                down_c = self.down_c(content)
                submoduled_s,submoduled_c,vq_loss_s,vq_loss_c = self.submodule(down_s , down_c)
                up_s = self.up(submoduled_s)
                up_c = self.up(submoduled_c)
                # 对于非最内层，添加skip连接和子模块
                return torch.cat([style, up_s], 1) , torch.cat([content , up_c],1),vq_loss_s,vq_loss_c