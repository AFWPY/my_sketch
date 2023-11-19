import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from .generator import Generator
from .discriminator import NLayerDiscriminator

import torchvision.models as models
from torchvision.transforms import ToPILImage
import torch.nn as nn
from .networks import GANLoss
import os

class CBGANModel(BaseModel):
    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        self.loss_names = ['loss_G_content', 'loss_G_style_rec','loss_D_content','vq_loss_content','vq_loss_style']
    
        self.gen = Generator(input_nc=3, output_nc=3,num_downs=16)
        self.dis = NLayerDiscriminator(3,64)

        if torch.cuda.device_count() > 1:
            self.gen = nn.DataParallel(self.gen,device_ids=[0, 1])
            self.dis = nn.DataParallel(self.dis,device_ids=[0, 1])
        
        self.gen.to(self.device)
        self.dis.to(self.device)

        self.criterionGAN = GANLoss('lsgan').to(self.device)
        self.criterionCycle = torch.nn.L1Loss().to(self.device)

        self.optimizer_G = torch.optim.Adam(itertools.chain(self.gen.parameters()), lr=0.0002, betas=(0.5, 0.999))
        self.optimizer_D = torch.optim.Adam(itertools.chain(self.dis.parameters()), lr=0.0002, betas=(0.5, 0.999))

        self.fake_content_pool = ImagePool(50)


    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        self.style= input['A'].to(self.device)
        self.content = input['B'].to(self.device)
        self.image_style = input['A_paths']
        self.image_content = input['B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_style,self.fake_content,self.vq_loss_style,self.vq_loss_content = self.gen(self.style,self.content)

    def backward_G(self):
        self.loss_G_content = self.criterionGAN(self.dis(self.fake_content), True)
        self.loss_G_style_rec = self.criterionCycle(self.fake_style, self.style)

        self.loss_G = self.vq_loss_content*100 + self.vq_loss_style*100 + self.loss_G_content*100 + self.loss_G_style_rec
        # 如果self.loss_G是一个向量而不是标量，那么对其取平均得到标量
        if self.loss_G.dim() > 0:  # 检查loss_G是否为标量
            self.loss_G = self.loss_G.mean()

        # print(f"loss_G : {self.loss_G.shape}")
        # 这时 self.loss_G 应该是一个标量，可以安全地调用 backward()
        self.loss_G.backward()


    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        if loss_D.dim() > 0:  # 检查loss_G是否为标量
            loss_D = loss_D.mean()
        loss_D.backward()
        return loss_D

    def backward_D_content(self):
        """Calculate GAN loss for discriminator D_A"""
        fake = self.fake_content_pool.query(self.fake_content)
        self.loss_D_content = self.backward_D_basic(self.dis, self.style, fake) # 判别生成的风格内容图和风格图

    def optimize_parameters(self,i):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad([self.dis], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights
        # if i%100==0:
        # # D_A and D_B
        self.set_requires_grad([self.dis], True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D_content()   # calculate graidents for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights
    
    def save_images(self,path,i):
        # 创建目录
        if not os.path.exists(path):
            os.makedirs(path)

        # 将张量转换为PIL图像
        image_s = ToPILImage()(self.fake_style[0].cpu())
        image_c = ToPILImage()(self.fake_content[0].cpu())
        

        # 生成文件名，确保将i转换为字符串
        filename_s = f'image_s_{i}.png'
        filename_c = f'image_c_{i}.png'

        # 构造完整的文件路径
        full_path_s = os.path.join(path, filename_s)
        full_path_c = os.path.join(path, filename_c)
        # 然后保存图像
        image_s.save(full_path_s)
        image_c.save(full_path_c)
    
    