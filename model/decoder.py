from functools import partial
import torch
import torch.nn as nn
from .modules import ConvBlock, ResBlock


class Integrator(nn.Module):
    def __init__(self, C, norm='none', activ='none', C_content=0, C_reference=0):
        super().__init__()
        C_in = C + C_content + C_reference
        self.integrate_layer = ConvBlock(C_in, C, 1, 1, 0, norm=norm, activ=activ)

    def forward(self, comps, content=None, reference=None):
        """
        Args:
            comps [B, 3, mem_shape]: component features
        """
        if reference==None:
            inputs = torch.concat((comps, content), 1)
            out = self.integrate_layer(inputs)
            return out
        else:
            inputs = torch.concat((comps, content, reference), 1)
            out = self.integrate_layer(inputs)
            return out


class Decoder(nn.Module):
    def __init__(self, layers, skips=None, out='sigmoid'):
        super().__init__()
        self.layers = nn.ModuleList(layers)

        if out == 'sigmoid':
            self.out = nn.Sigmoid()
        elif out == 'tanh':
            self.out = nn.Tanh()
        else:
            raise ValueError(out)

    def forward(self, x):
        """
        forward
        """
        for i, layer in enumerate(self.layers):
            x = layer(x)
        return self.out(x)



def dec_builder(C_in, C_out, norm='none', activ='relu', out='sigmoid'):

    ConvBlk = partial(ConvBlock, norm=norm, activ=activ)
    ResBlk = partial(ResBlock, norm=norm, activ=activ)

    layers = [
        ResBlk(C_in, C_in, 3, 1),
        ResBlk(C_in, C_in, 3, 1),
        ResBlk(C_in, C_in, 3, 1),
        ConvBlk(C_in, C_in//2, 3, 1, 1, upsample=True),  # 32x32
        ConvBlk(C_in//2, C_in//4, 3, 1, 1, upsample=True),  # 64x64
        ConvBlk(C_in//4, C_in//8, 3, 1, 1, upsample=True),  # 128x128
        ConvBlk(C_in//8, C_out, 3, 1, 1)
    ]

    return Decoder(layers, out=out)



