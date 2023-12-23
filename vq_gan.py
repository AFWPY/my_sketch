import torch
import torch.nn as nn
from networks import UnetSkipConnectionBlock,VectorQuantizerBlock,ResnetBlock,GANLoss
import functools
from util.image_pool import ImagePool
import itertools
import os
from torchvision.transforms import ToPILImage
import torch.nn.functional as F
import torchvision.models as models
import argparse
from dataset import CustomDataset
from torch.utils.data import Dataset, DataLoader
import time
import shutil
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
        embedding_dim = 64*64 # 512*1*1
        commitment_cost = 0.25
        self.vq = VectorQuantizerBlock(num_embeddings, embedding_dim, commitment_cost)

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        down_s = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]
        
        down_p = [nn.ReflectionPad2d(3),
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
        
    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
                    
    def forward(self, input_p,input_s):
        """Standard forward"""
        down_p = self.down_p(input_p)
        down_s = self.down_s(input_s)

        vq_p,vqloss_p = self.vq(down_p,key = "Content")
        vq_s,vqloss_s = self.vq(down_s)

        res_p = self.ResBlocks(vq_p)
        res_s = self.ResBlocks(vq_s)

        up_p = self.up(res_p)
        up_s = self.up(res_s)
        return up_p,up_s,vqloss_p,vqloss_s

class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)
    
class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.gen = ResnetGenerator(input_nc=3, output_nc=3,n_blocks=9).to(self.device)
        self.dis = NLayerDiscriminator(3,64).to(self.device)

#         if torch.cuda.device_count() > 1:
#             self.gen = nn.DataParallel(self.gen,device_ids=[0, 1])
#             self.dis = nn.DataParallel(self.dis,device_ids=[0, 1])
        
        self.criterionGAN = GANLoss('lsgan').to(self.device)
        self.criterionCycle = torch.nn.L1Loss().to(self.device)
        self.criterionRec = nn.MSELoss().to(self.device)


        #使用vgg19特征层
        vgg19 = models.vgg19(pretrained=True)
        self.vgg_f = vgg19.features.to(self.device)# 特征层
        

        self.optimizer_G = torch.optim.Adam(itertools.chain(self.gen.parameters()), lr=0.0002, betas=(0.5, 0.999))
        self.optimizer_D = torch.optim.Adam(itertools.chain(self.dis.parameters()), lr=0.0002, betas=(0.5, 0.999))

        self.fake_content_pool = ImagePool(50)

    # 定义Gram矩阵计算函数
    def gram_matrix(self,feature):
        batch_size, channels, height, width = feature.size()
        feature_reshaped = feature.view(batch_size, channels, -1)
        gram = torch.bmm(feature_reshaped, feature_reshaped.transpose(1, 2))
        gram = gram / (channels * height * width)
        return gram
    
    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        self.photo= input['A'].to(self.device)
        self.sketch = input['B'].to(self.device)
        self.sketch_r = input['C'].to(self.device)
        self.sketch_rt = input['CT'].to(self.device)
        self.image_p = input['A_paths']
        self.image_s = input['B_paths']
        self.image_sr = input['C_paths']


    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_p,self.fake_s,self.vq_loss_p,self.vq_loss_s = self.gen(self.photo,self.sketch)

    def backward_G(self,i):
        #由照片生成的图片和判别器的对抗损失
        if(i<50000):
            GM_p = 0.1
        else:
            GM_p = 10
        self.loss_G_content = self.criterionGAN(self.dis(self.fake_p), True)
        # #由照片生成的图片和原图片的使用vgg19特征对比损失
        self.loss_precs = self.criterionRec((self.vgg_f(self.fake_p)), (self.vgg_f(self.sketch)))
        #由照片生成的图片还原损失
        self.loss_G_content_rec = self.criterionRec(self.fake_p, self.sketch)
        #由素描生成的图片还原损失
        self.loss_G_style_rec = self.criterionRec(self.fake_s, self.sketch)
        
        
        self.loss_photo = self.loss_precs + self.loss_G_content + self.loss_G_content_rec*1000 + self.vq_loss_p
        self.loss_sketch = self.loss_G_style_rec*1000 + self.vq_loss_s*10

        self.loss_G = self.loss_photo*GM_p + self.loss_sketch*10
        # 如果self.loss_G是一个向量而不是标量，那么对其取平均得到标量
        if self.loss_G.dim() > 0:  # 检查loss_G是否为标量
            self.loss_G = self.loss_G.mean()

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
        fake = self.fake_content_pool.query(self.fake_p)
        self.loss_D_content = self.backward_D_basic(self.dis, self.sketch, fake) # 判别生成的风格内容图和风格图

    def optimize_parameters(self,i):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad([self.dis], False)  # Ds require no gradients when optimizing Gs
        if (i<50000):
            self.set_requires_grad([self.gen.down_p], False)
        else:
            self.set_requires_grad([self.gen.down_p], True)
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G(i)             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights
        
        self.set_requires_grad([self.dis], True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D_content()   # calculate graidents for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights
    
    def save_images(self,path,i):
        # 创建目录
        if not os.path.exists(path):
            os.makedirs(path)

        # 将张量转换为PIL图像
        image_s = ToPILImage()(self.fake_s[0].cpu())
        image_p = ToPILImage()(self.fake_p[0].cpu())
        

        # 生成文件名，确保将i转换为字符串
        filename_s = f'image_s_{i}.png'
        filename_p = f'image_p_{i}.png'

        # 构造完整的文件路径
        full_path_s = os.path.join(path, filename_s)
        full_path_p = os.path.join(path, filename_p)
        # 然后保存图像
        image_s.save(full_path_s)
        image_p.save(full_path_p)
        
    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
                    
    def get_current_losses(self,epoch,log_name):
        content = f'epoch:{epoch}  loss_G_content:{self.loss_G_content}  loss_precs:{self.loss_precs} loss_G_style_rec:{self.loss_G_style_rec} loss_G_content_rec:{self.loss_G_content_rec}'
        with open(log_name,'a') as file:
            file.write(content)
    
                    
def setup_opt():
    """
    setup_opt
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')
    parser.add_argument("--resume", default=None, help="path/to/saved/.pth")
    parser.add_argument('--dataroot', required=True, help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--num_threads', default=32, type=int, help='# threads for loading data')
    parser.add_argument('--batch_size', type=int, default=8, help='input batch size')
    parser.add_argument('--load_size', type=int, default=286, help='scale images to this size')
    parser.add_argument('--crop_size', type=int, default=256, help='then crop to this size')
    parser.add_argument('--dataset_mode', type=str, default='unaligned', help='chooses how datasets are loaded. [unaligned | aligned | single | colorization]')
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
    parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
    parser.add_argument('--preprocess', type=str, default='resize_and_crop', help='scaling and cropping of images at load time [resize_and_crop | crop | scale_width | scale_width_and_crop | none]')
    parser.add_argument('--n_epochs', type=int, default=200000, help='number of epochs with the initial learning rate')
    
    opt = parser.parse_args()

    return opt
                    
if __name__ == "__main__":
    
    opt = setup_opt()
    # 创建数据集
    train_data_root = "./data/CUHK/"
    dataset = CustomDataset(root=train_data_root)

    # 创建数据加载器
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    model = model()
    
    # 确保这里的目录结构存在
    log_dir = os.path.join(opt.checkpoints_dir, opt.name)
    shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)
    
    # 文件路径
    log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
    # 检查文件是否存在，如果不存在则创建
    if not os.path.exists(log_name):
        with open(log_name, 'w') as log_file:
            pass
    image_path = os.path.join(opt.checkpoints_dir, opt.name)

    for epoch in range(opt.n_epochs):
        if(epoch % 100 ==0):
            epoch_start_time = time.time()  # timer for entire epoch
        # model.update_learning_rate()
        
        for data in dataloader:
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters(epoch)

        if epoch % 1000 == 0:
            model.save_images(image_path,epoch) 
            
             # 打印loss
            model.get_current_losses(epoch,log_name)

        # 保存模型
        if epoch % 10000 == 0:
            torch.save(model.state_dict(), f'{opt.checkpoints_dir}/{opt.name}/model_weights_epoch_{epoch}.pth')
        


