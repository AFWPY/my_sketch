import traceback
import torch
import torch.nn as nn
import torch.nn.functional as F
import functools

from model.decoder import dec_builder
from model.content_encoder import content_enc_builder
from model.references_encoder import comp_enc_builder
from model.Component_Attention_Module import ComponentAttentiomModule
from model.memory import Memory
from model.VectorQuantizer import  VectorQuantizer,VectorQuantizerEMA


class Generator(nn.Module):
    """
    Generator
    """

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(Generator,self).__init__()
    #     # 风格编码器
    #     self.styleGen = comp_enc_builder(C_in, C, C_out) # B*C_in*256*256 -> B*C_out*32*32

    #     # 内容编码器
    #     self.contentGen = content_enc_builder(C_in, C, C_out) # B*C_in*256*256 -> B*C_out*32*32

        # codebook使用VQ-VAE进行编码
        num_embeddings = 512 # 嵌入向量数量，过多容易过拟合，过少容易欠拟合
        embedding_dim = 1024 # 32*32 
        commitment_cost = 0.25
        self.Codebook = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)

    #     # MHA多头注意力机制，输入style生成的KV,CodeBook生成的Q
    #     self.MHA = ComponentAttentiomModule()

    #     # 解码器
    #     self.Decoder = dec_builder(C_out,C_in)  # B*C_out*32*32 -> B*C_in*256*256

    
    # def get_style_matrix(self,input):
    #     input = self.styleGen(input)
    #     query_cb,style_cb_loss = self.Codebook(input)
    #     style_matrix = self.MHA(input,query_cb)

    #     return style_matrix,style_cb_loss
    
    # def get_content_matrix(self,input):
    #     input = self.contentGen(input)
    #     query_cb,content_cb_loss = self.Codebook(input)
    #     content_matrix = self.MHA(input,query_cb)

    #     return content_matrix,content_cb_loss
        
    # def decode(self,input):
    #     fake_img = self.Decoder(input)
    #     return fake_img


# construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)

class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
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
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)