import torch
import torch.nn as nn
import itertools
from .networks import Generator, Discriminator, init_weights, get_norm_layer


class CycleGANModel:
    """CycleGAN模型类"""
    def __init__(self, device, lr=0.0002, beta1=0.5, lambda_A=10.0, lambda_B=10.0, lambda_identity=0.5):
        """
        参数:
            device: 训练设备 (cuda/cpu)
            lr: 学习率
            beta1: Adam优化器的beta1参数
            lambda_A: A域循环一致性损失权重
            lambda_B: B域循环一致性损失权重
            lambda_identity: 身份映射损失权重
        """
        self.device = device
        self.lambda_A = lambda_A
        self.lambda_B = lambda_B
        self.lambda_identity = lambda_identity

        norm_layer = get_norm_layer(norm_type='instance')

        # 创建生成器 (针对8GB显存优化: 使用6个ResNet块)
        self.netG_A2B = Generator(input_nc=3, output_nc=3, ngf=64, n_blocks=6, norm_layer=norm_layer).to(device)
        self.netG_B2A = Generator(input_nc=3, output_nc=3, ngf=64, n_blocks=6, norm_layer=norm_layer).to(device)

        # 创建判别器
        self.netD_A = Discriminator(input_nc=3, ndf=64, n_layers=3, norm_layer=norm_layer).to(device)
        self.netD_B = Discriminator(input_nc=3, ndf=64, n_layers=3, norm_layer=norm_layer).to(device)

        # 初始化权重
        init_weights(self.netG_A2B, init_type='normal', init_gain=0.02)
        init_weights(self.netG_B2A, init_type='normal', init_gain=0.02)
        init_weights(self.netD_A, init_type='normal', init_gain=0.02)
        init_weights(self.netD_B, init_type='normal', init_gain=0.02)

        # 定义损失函数
        self.criterionGAN = nn.MSELoss()
        self.criterionCycle = nn.L1Loss()
        self.criterionIdentity = nn.L1Loss()

        # 定义优化器
        self.optimizer_G = torch.optim.Adam(
            itertools.chain(self.netG_A2B.parameters(), self.netG_B2A.parameters()),
            lr=lr, betas=(beta1, 0.999)
        )
        self.optimizer_D = torch.optim.Adam(
            itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()),
            lr=lr, betas=(beta1, 0.999)
        )

        # 图像缓冲区，用于更新判别器
        self.fake_A_buffer = ImageBuffer()
        self.fake_B_buffer = ImageBuffer()

    def set_input(self, real_A, real_B):
        """设置输入图像"""
        self.real_A = real_A.to(self.device)
        self.real_B = real_B.to(self.device)

    def forward(self):
        """前向传播"""
        self.fake_B = self.netG_A2B(self.real_A)  # G_A2B(A)
        self.rec_A = self.netG_B2A(self.fake_B)   # G_B2A(G_A2B(A))
        self.fake_A = self.netG_B2A(self.real_B)  # G_B2A(B)
        self.rec_B = self.netG_A2B(self.fake_A)   # G_A2B(G_B2A(B))

    def backward_D_basic(self, netD, real, fake):
        """计算判别器的基本损失"""
        # 真实图像
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, torch.ones_like(pred_real))

        # 假图像
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, torch.zeros_like(pred_fake))

        # 总损失
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """计算判别器D_A的损失"""
        fake_B = self.fake_B_buffer.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_B, self.real_B, fake_B)

    def backward_D_B(self):
        """计算判别器D_B的损失"""
        fake_A = self.fake_A_buffer.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_A, self.real_A, fake_A)

    def backward_G(self):
        """计算生成器的损失"""
        # 身份损失
        if self.lambda_identity > 0:
            # G_A2B应该将B映射为B的身份
            self.idt_A = self.netG_A2B(self.real_B)
            self.loss_idt_A = self.criterionIdentity(self.idt_A, self.real_B) * self.lambda_B * self.lambda_identity
            # G_B2A应该将A映射为A的身份
            self.idt_B = self.netG_B2A(self.real_A)
            self.loss_idt_B = self.criterionIdentity(self.idt_B, self.real_A) * self.lambda_A * self.lambda_identity
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN损失 D_B(G_A2B(A))
        self.loss_G_A2B = self.criterionGAN(self.netD_B(self.fake_B), torch.ones_like(self.netD_B(self.fake_B)))
        # GAN损失 D_A(G_B2A(B))
        self.loss_G_B2A = self.criterionGAN(self.netD_A(self.fake_A), torch.ones_like(self.netD_A(self.fake_A)))

        # 前向循环损失 ||G_B2A(G_A2B(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * self.lambda_A
        # 后向循环损失 ||G_A2B(G_B2A(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * self.lambda_B

        # 总生成器损失
        self.loss_G = self.loss_G_A2B + self.loss_G_B2A + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B
        self.loss_G.backward()

    def optimize_parameters(self):
        """计算损失、梯度，并更新网络权重"""
        # 前向传播
        self.forward()

        # 更新生成器 G_A2B 和 G_B2A
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

        # 更新判别器 D_A 和 D_B
        self.optimizer_D.zero_grad()
        self.backward_D_A()
        self.backward_D_B()
        self.optimizer_D.step()

    def get_current_losses(self):
        """返回当前训练损失"""
        losses = {
            'D_A': self.loss_D_A.item(),
            'D_B': self.loss_D_B.item(),
            'G_A2B': self.loss_G_A2B.item(),
            'G_B2A': self.loss_G_B2A.item(),
            'Cycle_A': self.loss_cycle_A.item(),
            'Cycle_B': self.loss_cycle_B.item(),
        }
        if self.lambda_identity > 0:
            losses['Idt_A'] = self.loss_idt_A.item()
            losses['Idt_B'] = self.loss_idt_B.item()
        return losses

    def save_networks(self, epoch, save_dir):
        """保存模型"""
        torch.save(self.netG_A2B.state_dict(), f'{save_dir}/netG_A2B_epoch_{epoch}.pth')
        torch.save(self.netG_B2A.state_dict(), f'{save_dir}/netG_B2A_epoch_{epoch}.pth')
        torch.save(self.netD_A.state_dict(), f'{save_dir}/netD_A_epoch_{epoch}.pth')
        torch.save(self.netD_B.state_dict(), f'{save_dir}/netD_B_epoch_{epoch}.pth')

    def load_networks(self, epoch, save_dir):
        """加载模型"""
        self.netG_A2B.load_state_dict(torch.load(f'{save_dir}/netG_A2B_epoch_{epoch}.pth'))
        self.netG_B2A.load_state_dict(torch.load(f'{save_dir}/netG_B2A_epoch_{epoch}.pth'))
        self.netD_A.load_state_dict(torch.load(f'{save_dir}/netD_A_epoch_{epoch}.pth'))
        self.netD_B.load_state_dict(torch.load(f'{save_dir}/netD_B_epoch_{epoch}.pth'))


class ImageBuffer:
    """
    图像缓冲区，用于存储之前生成的图像
    从缓冲区返回图像的概率为50%，返回当前批次图像的概率为50%
    """
    def __init__(self, pool_size=50):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = torch.rand(1).item()
                if p > 0.5:
                    random_id = torch.randint(0, self.pool_size, (1,)).item()
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image)
        return_images = torch.cat(return_images, 0)
        return return_images
