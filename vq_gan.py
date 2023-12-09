import torch.nn as nn
# from model.Component_Attention_Module import ComponentAttentionModule
from networks import UnetSkipConnectionBlock,VectorQuantizerBlock,ResnetBlock
import functools
class UnetGenerator(nn.Module):
    """
    Generator
    """

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetGenerator,self).__init__()

        # codebook使用VQ-VAE进行编码
        num_embeddings = 8192 # 嵌入向量数量，过多容易过拟合，过少容易欠拟合
        embedding_dim = 512*1*1 # 512*1*1
        commitment_cost = 0.25
        self.vq = VectorQuantizerBlock(num_embeddings, embedding_dim, commitment_cost)

        # MHA多头注意力机制，输入input和vq生成的Query
        # self.MHA = ComponentAttentionModule()

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=self.vq, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input),self.vq.loss
    

class ResnetGenerator(nn.Module):
    """
    Generator
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        super(ResnetGenerator,self).__init__()

        # codebook使用VQ-VAE进行编码
        num_embeddings = 8192 # 嵌入向量数量，过多容易过拟合，过少容易欠拟合
        embedding_dim = 512*1*1 # 512*1*1
        commitment_cost = 0.25
        self.vq = VectorQuantizerBlock(num_embeddings, embedding_dim, commitment_cost)

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        down_p = down_s = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2

        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            down_p += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            down_s += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        ResBlocks = []
        for i in range(n_blocks):       # add ResNet blocks
            ResBlocks += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        up = []
        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            up += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        up += [nn.ReflectionPad2d(3)]
        up += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        up += [nn.Tanh()]

        self.down_p = nn.Sequential(*down_p)
        self.down_s = nn.Sequential(*down_s)
        self.up = nn.Sequential(*up)
        self.ResBlocks = nn.Sequential(*ResBlocks)
    def forward(self, input_p,input_s):
        """Standard forward"""
        down_p = self.down_p(input_p)
        down_s = self.down_s(input_s)

        vq_p,vqloss_p = self.vq(down_p,key = "Context")
        vq_s,vqloss_s = self.vq(down_s)

        res_p = self.ResBlocks(vq_p)
        res_s = self.ResBlocks(vq_s)

        up_p = self.up(res_p)
        up_s = self.up(res_s)
        return up_p,up_s,vqloss_p,vqloss_s


