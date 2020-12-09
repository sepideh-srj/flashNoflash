import torch
from .base_model import BaseModel
from . import networks
import matplotlib.pyplot as plt
import numpy
import itertools
from util.image_pool import ImagePool
import kornia
from torchvision import transforms

class CyclePix2PixLabModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='aligned')
        parser.add_argument('--ratio_rec', type=float, default=0, help='if 1 rec = ratio_rec, if 0 rec = real rec')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=1000.0, help='weight for L1 loss')
            parser.add_argument('--lambda_A', type=float, default=1000.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=1000.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_comp', type=float, default=0, help='')
            parser.add_argument('--cycle_epoch', type=float, default=30, help='')
            parser.add_argument('--D_flash', type= float, default=0)

            # parser.add_argument('--ratio_rec', type=float, default=0, help='if 1 rec = ratio_rec, if 0 rec = real rec')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN_A', 'G_L1_A', 'D_A', 'G_GAN_B', 'G_L1_B', 'D_B', 'cycle_B', 'cycle_A', 'G_L1_A_comp', 'G_L1_B_comp', 'D_A_F', 'D_B_F', 'G_GAN_flash_A', 'G_GAN_flash_B','G_GAN_recB', 'G_GAN_recA']
        visual_names_A = ['real_A', 'fake_A_output']
        visual_names_B = ['real_B', 'fake_B_output', 'real_C']
        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B', 'D_A_F', 'D_B_F']
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B']
        # define networks (both generator and discriminator)
            # define networks (both generator and discriminator)
        if self.opt.midas:
            self.netG_A = networks.define_G(opt.input_nc + 1, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                            not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netG_B = networks.define_G(opt.output_nc + 1, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                            not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        else:
            self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                            not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                            not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        if self.isTrain:  # define discriminators
            print("kiiiir")
            self.netD_B = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_A = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            # if self.opt.D_flash:
            #     print("kir")
            self.netD_B_F = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_A_F = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionCycle = torch.nn.L1Loss()
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters(), self.netD_A_F.parameters(), self.netD_B_F.parameters()),
                                                    lr=opt.lr2, betas=(opt.beta1, 0.999))
            # else:
            #     self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.lr2, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_A_copy = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']


        self.real_C = self.real_A.detach().clone() - self.real_B.detach().clone()
        self.real_C = (((self.real_C - torch.min(self.real_C))/(torch.max(self.real_C) - torch.min(self.real_C))) - 0.5)*2
        if self.opt.midas:
            self.midas_A = input['midas_A' if AtoB else 'midas_B'].to(self.device)
            self.midas_B = input['midas_B' if AtoB else 'midas_A'].to(self.device)
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        if self.opt.midas:
            self.A_midas = torch.cat((self.real_A, self.midas_A), 1)
            fake_B = self.netG_A(self.A_midas)  # G_A(A)
            self.fake_B_output = (2 * (self.real_A * fake_B + 2 * fake_B + self.real_A) - 1) / 5
            self.fake_B_output_midas = torch.cat((self.fake_B_output, self.midas_A), 1)
            self.rec_A = self.netG_B(self.fake_B_output_midas)  # G_B(G_A(A))
            # self.rec_A = (2 * (self.fake_B_output * self.rec_A_ratio + 2 * self.rec_A_ratio + self.fake_B_output) - 1) / 5

            self.real_B_midas = torch.cat((self.real_B, self.midas_B), 1)
            self.fake_A = self.netG_B(self.real_B_midas)  # G_B(B)
            self.fake_A_output = (2 * (self.real_B * self.fake_A + 2 * self.fake_A + self.real_B) - 1) / 5
            self.fake_A_output_midas = torch.cat((self.fake_A_output, self.midas_B), 1)
            self.rec_B = self.netG_A(self.fake_A_output_midas)  # G_A(G_B(B))
            # self.rec_B_ratio = (2 * (self.fake_A_output * self.rec_B_ratio + 2 * self.rec_B_ratio + self.fake_A_output) - 1) / 5
        else:
            fake_B = self.netG_A(self.real_A)  # G_A(A)
            self.fake_B_output = (2*(self.real_A*fake_B+ 2*fake_B+ self.real_A) - 1) / 5
            # if self.opt.ratio_rec:
            #     self.rec_A = self.netG_B(self.fake_B_output)  # G_B(G_A(A))
            # else:
            rec_A_ratio = self.netG_B(self.fake_B_output)
            self.rec_A = (2 * (self.fake_B_output * rec_A_ratio + 2 * rec_A_ratio + self.fake_B_output) - 1) / 5
            fake_A = self.netG_B(self.real_B)  # G_B(B)
            self.fake_A_output = (2*(self.real_B*fake_A+ 2*fake_A+ self.real_B) - 1) / 5

            # self.real_ratio = ((2 * (self.real_A + 1) / (self.real_B + 2)) - 0.8) * 5 / 4
            # self.real_ratio = ((self.real_A + 1) / (self.real_B + 2)*4) - (1)
            # if self.opt.ratio_rec:
            #     self.rec_B = self.netG_A(self.fake_A_output)  # G_A(G_B(B))
            # else:
            rec_B_ratio = self.netG_A(self.fake_A_output)
            self.rec_B = (2 * (self.fake_A_output * rec_B_ratio + 2 * rec_B_ratio + self.fake_A_output) - 1) / 5

    def backward_D_basic(self, netD, real_A, real_B, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Fake
        fake_AB = torch.cat((real_A, fake), 1)
        pred_fake = netD(fake_AB.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_AB = torch.cat((real_A, real_B), 1)
        pred_real = netD(real_AB)
        loss_D_real = self.criterionGAN(pred_real, True)

        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        # fake_B = self.fake_B_pool.query(self.fake_B_output)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_A, self.real_B, self.fake_B_output)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        # fake_A = self.fake_B_pool.query(self.fake_A_output)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_B, self.real_A, self.fake_A_output)
    def backward_D_A_F(self):
        self.loss_D_A_F = self.backward_D_basic(self.netD_A_F, self.real_A, self.real_C, self.A_comp)
    def backward_D_B_F(self):
        self.loss_D_B_F = self.backward_D_basic(self.netD_B_F, self.real_B, self.real_C, self.B_comp)

    def backward_G(self,epoch):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B_output), 1)
        pred_fake = self.netD_A(fake_AB)
        self.loss_G_GAN_A = self.criterionGAN(pred_fake, True)
        fake_AB_B = torch.cat((self.real_B, self.fake_A_output), 1)
        pred_fake = self.netD_B(fake_AB_B)
        self.loss_G_GAN_B = self.criterionGAN(pred_fake, True)

        fake_AB = torch.cat((self.real_A, self.rec_B), 1)
        pred_fake = self.netD_A(fake_AB)
        self.loss_G_GAN_recA = self.criterionGAN(pred_fake, True)
        fake_AB_B = torch.cat((self.real_B, self.rec_A), 1)
        pred_fake = self.netD_B(fake_AB_B)
        self.loss_G_GAN_recB = self.criterionGAN(pred_fake, True)
        self.loss_G_GAN_flash_B = 0
        self.loss_G_GAN_flash_A = 0
        if self.opt.D_flash and epoch >10:
            fake_AC = torch.cat((self.real_A, self.A_comp), 1)
            pred_fake = self.netD_A_F(fake_AC)
            self.loss_G_GAN_flash_A = self.criterionGAN(pred_fake, True)
            fake_BC = torch.cat((self.real_B, self.B_comp), 1)
            pred_fake = self.netD_B_F(fake_BC)
            self.loss_G_GAN_flash_B = self.criterionGAN(pred_fake, True)

        # Second, G(A) = B
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = 0
        self.loss_cycle_B = 0
        if self.opt.lambda_comp !=0:
            A = self.fake_A_output.detach().clone()
            B = self.real_B.detach().clone()
            C = A - B
            fake_C_A = (((C - torch.min(C)) / (
                        torch.max(C) - torch.min(C))) - 0.5) * 2

            self.A_comp = fake_C_A
            A = self.real_A.detach().clone()
            B = self.fake_B_output.detach().clone()
            C = A - B
            fake_C_B = (((C - torch.min(C)) / (
                    torch.max(C) - torch.min(C))) - 0.5) * 2


            self.B_comp = fake_C_B

            # Acomp = real_A - fake_B_output
            self.loss_G_L1_A_comp = self.criterionL1(self.A_comp, self.real_C) * self.opt.lambda_comp
            # Bcomp = fake_A_output - real_B
            self.loss_G_L1_B_comp = self.criterionL1(self.B_comp, self.real_C) * self.opt.lambda_comp
        else:
            self.loss_G_L1_A_comp = 0
            self.loss_G_L1_B_comp = 0
        if epoch >= self.opt.cycle_epoch:
            self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * self.opt.lambda_A
            # Backward cycle loss || G_A(G_B(B)) - B||
            self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * self.opt.lambda_B
        self.loss_G_L1_A = self.criterionL1(self.fake_A_output, self.real_A) * self.opt.lambda_L1
        self.loss_G_L1_B = self.criterionL1(self.fake_B_output, self.real_B) * self.opt.lambda_L1
        # combine loss and calculate gradients
        self.loss_G =self.loss_G_GAN_flash_A + self.loss_G_GAN_flash_B + self.loss_G_GAN_recA + self.loss_G_GAN_recB + self.loss_G_GAN_B + self.loss_G_GAN_A + self.loss_cycle_A + self.loss_cycle_B + self.loss_G_L1_A+ self.loss_G_L1_B + self.loss_G_L1_A_comp + self.loss_G_L1_B_comp
        self.loss_G.backward()

    def optimize_parameters(self, epoch):
        self.forward()  # compute fake images: G(A)
        # update D
        if self.opt.D_flash:
            print("here")
            self.set_requires_grad([self.netD_A, self.netD_B, self.netD_A_F, self.netD_B_F], True)  # enable backprop for D
            self.optimizer_D.zero_grad()  # set D's gradients to zero
            self.backward_D_A()  # calculate gradients for D_A
            self.backward_D_B()
            if epoch > 10:
                self.backward_D_A_F()
                self.backward_D_B_F()
            else:
                self.loss_D_A_F = 1
                self.loss_D_B_F = 1
            self.optimizer_D.step()  # update D's weights
            # update G
            self.set_requires_grad([self.netD_A, self.netD_B, self.netD_A_F, self.netD_B_F], False)  # D requires no gradients when optimizing G
            self.optimizer_G.zero_grad()  # set G's gradients to zero
            self.backward_G(epoch)  # calculate graidents for G
            self.optimizer_G.step()
        else:
            self.set_requires_grad([self.netD_A, self.netD_B], True)  # enable backprop for D
            self.optimizer_D.zero_grad()  # set D's gradients to zero
            self.backward_D_A()      # calculate gradients for D_A
            self.backward_D_B()
            self.optimizer_D.step()  # update D's weights
            # update G
            self.set_requires_grad([self.netD_A, self.netD_B], False)  # D requires no gradients when optimizing G
            self.optimizer_G.zero_grad()  # set G's gradients to zero
            self.backward_G(epoch)  # calculate graidents for G
            self.optimizer_G.step()  # udpate G's weights

    def showImage(self, image):
        # image = torch.reshape(image, (256, 256, 3))
        img = image.cpu()
        img = torch.squeeze(img)
        img = img.numpy()
        # output = img[0]
        output = (numpy.transpose(img, (1, 2, 0)) + 1) / 2.0 * 255.0
        print(output.shape)
        output = numpy.where(output > 255, 255, output)
        output = (output).astype(numpy.uint8)
        print(numpy.max(output))
        plt.imshow(output)
        plt.colorbar()
        plt.show()

