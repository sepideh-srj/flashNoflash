import torch
from .base_model import BaseModel
from . import networks
import matplotlib.pyplot as plt
import numpy as np
import numpy
import itertools
from util.image_pool import ImagePool
from torchvision import transforms
import cv2
import kornia

class SharedDecModel(BaseModel):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.add_argument('--ratio', type=float, default=1)
        parser.add_argument('--lambda_comp', type=float, default=0, help='')
        parser.add_argument('--lambda_color_uv', type=float, default=0, help='')
        parser.add_argument('--D_flash', type= float, default=0)
        parser.add_argument('--dslr_color_loss', type=float, default=0)
        parser.add_argument('--dec_features_num', type=float, default=32, help='')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
            parser.add_argument('--lambda_A', type=float, default=25.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=25.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--cycle_epoch', type=float, default=30, help='')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN_A', 'G_L1_A', 'G_GAN_B', 'G_L1_B', 'cycle_B', 'cycle_A', 'G_L1_A_comp', 'G_L1_B_comp', 'G_GAN_flash_A', 'G_GAN_flash_B','G_GAN_recB', 'G_GAN_recA', 'G_L1_A_comp_color', 'G_L1_B_comp_color', 'color_dslr_B', 'color_dslr_A']
        if self.opt.D_flash:
            self.loss_names += ['D_F']
        else:
            self.loss_names += ['D_A', 'D_B']
        visual_names_B = ['real_A', 'fake_A']
        visual_names_A = ['real_B', 'fake_B']
        self.visual_names = visual_names_B + visual_names_A #+ visual_names_C  # combine visualizations for A and B

        self.model_names = ['G_Decompostion', 'G_Generation','F_Decoder']

        if self.isTrain:
            if self.opt.D_flash:
                self.model_names += ['D_Flash']
            else:
                self.model_names += ['D_Decompostion', 'D_Generation']


        if self.opt.midas:
            self.netF_Decoder = networks.define_G(opt.input_nc+1, opt.dec_features_num, int(opt.dec_features_num*2), 'resnet_12blocks', opt.norm,
                                                       not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        else:
            self.netF_Decoder = networks.define_G(opt.input_nc, opt.dec_features_num, int(opt.dec_features_num*2), 'resnet_12blocks',opt.norm,
                                                  not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        self.netG_Decompostion = networks.define_G(opt.input_nc + opt.dec_features_num, opt.output_nc, opt.ngf,
                                                   'resnet_6blocks', opt.norm,
                                                   not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_Generation = networks.define_G(opt.input_nc + opt.dec_features_num, opt.output_nc, opt.ngf,
                                                 'resnet_6blocks', opt.norm,
                                                 not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            if self.opt.D_flash:
                self.netD_Flash = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                                    opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            else:
                self.netD_Decompostion = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                                           opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
                self.netD_Generation = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                                         opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)


        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionCycle = torch.nn.L1Loss()

            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_Decompostion.parameters(), self.netG_Generation.parameters(), self.netF_Decoder.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizers.append(self.optimizer_G)

            if self.opt.D_flash:
                self.optimizer_D1 = torch.optim.Adam(itertools.chain(self.netD_Flash.parameters()), lr=opt.lr3, betas=(opt.beta1, 0.999))
                self.optimizer_D2 = torch.optim.Adam(itertools.chain(self.netD_Flash.parameters()), lr=opt.lr3, betas=(opt.beta1, 0.999))
                self.optimizers.append(self.optimizer_D1)
                self.optimizers.append(self.optimizer_D2)
            else:
                self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_Decompostion.parameters(), self.netD_Generation.parameters()), lr=opt.lr3, betas=(opt.beta1, 0.999))
                self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        self.real_A = input['A'].to(self.device)
        self.real_B = input['B'].to(self.device)
        self.image_paths = input['A_paths']

        self.real_C = self.real_A - self.real_B

        if self.opt.midas:
            self.midas_A = input['depth_A'].to(self.device)
            self.midas_B = input['depth_B'].to(self.device)


    def applyratio(self,input,ratio):
        output = (3*input*ratio + 9*ratio + 5*input + 3) / 4
        return output


    def forward_onedirection(self,real_A,real_B,midas_A,midas_B):
        ## Adding depth if needed
        if self.opt.midas:
            decomposition_input = torch.cat((real_A, midas_A), 1)
            generation_input = torch.cat((real_B, midas_B), 1)
        else:
            decomposition_input = real_A
            generation_input = real_B

        ## forward into networks

        decomposition_features = self.netF_Decoder(decomposition_input)
        generation_features = self.netF_Decoder(generation_input)

        decomposition_features_and_input = torch.cat((real_A, decomposition_features), 1)
        generation_features_and_input = torch.cat((real_B, generation_features), 1)

        decomposition_output = self.netG_Decompostion(decomposition_features_and_input)
        generation_output = self.netG_Generation(generation_features_and_input)

        ## applying ratio if needed
        if self.opt.ratio:
            fake_B = self.applyratio(self.real_A,decomposition_output)
            fake_A = self.applyratio(self.real_B,generation_output)
        else:
            fake_B = decomposition_output
            fake_A = generation_output

        return fake_A, fake_B

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""

        self.fake_A, self.fake_B = self.forward_onedirection(self.real_A, self.real_B, self.midas_A, self.midas_B)

        ## compute estimated flash
        self.flash_from_decomposition = self.real_A - self.fake_B
        self.flash_from_generation = self.fake_A - self.real_B

        ##### Cycle PATH
        self.rec_A, self.rec_B = self.forward_onedirection(self.fake_A, self.fake_B, self.midas_B, self.midas_A)



    def forward_bilateral(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        if self.opt.midas:
            A_midas = torch.cat((self.real_A, self.midas_A), 1)
            self.fake_B = self.netG_Decompostion(A_midas)  # G_A(A)

            B_midas = torch.cat((self.real_B, self.midas_B), 1)
            self.fake_A = self.netG_Generation(B_midas)  # G_B(B)
        else:
            self.fake_B = self.netG_Decompostion(self.real_A)  # G_A(A)
            self.fake_A = self.netG_Generation(self.real_B)  # G_B(B)

    def backward_D_basic(self, netD, real_A, real_B, fake):
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
        self.loss_D_A = self.backward_D_basic(self.netD_Decompostion, self.real_A, self.real_B, self.fake_B)

    def backward_D_B(self):
        self.loss_D_B = self.backward_D_basic(self.netD_Generation, self.real_B, self.real_A, self.fake_A)

    def backward_D_F_1(self):
        self.loss_D_F = self.backward_D_basic(self.netD_Flash, self.real_B, self.real_C, self.flash_from_generation)

    def backward_D_F_2(self):
        self.loss_D_F = self.backward_D_basic(self.netD_Flash, self.real_A, self.real_C, self.flash_from_decomposition)


    def backward_G(self,epoch):

        self.loss_G_GAN_A = 0
        self.loss_G_GAN_B = 0
        self.loss_G_GAN_flash_B = 0
        self.loss_G_GAN_flash_A = 0
        self.loss_G_GAN_recA = 0
        self.loss_G_GAN_recB = 0


        self.loss_cycle_A = 0
        self.loss_cycle_B = 0
        self.loss_G_L1_A_comp = 0
        self.loss_G_L1_B_comp = 0
        self.loss_G_L1_A_comp_color = 0
        self.loss_G_L1_B_comp_color = 0
        self.loss_color_dslr_A = 0
        self.loss_color_dslr_B = 0

        if self.opt.D_flash:
            fake_AC = torch.cat((self.real_A, self.flash_from_decomposition), 1)
            pred_fake = self.netD_Flash(fake_AC)
            self.loss_G_GAN_flash_A = self.criterionGAN(pred_fake, True)
            fake_BC = torch.cat((self.real_B, self.flash_from_generation), 1)
            pred_fake = self.netD_Flash(fake_BC)
            self.loss_G_GAN_flash_B = self.criterionGAN(pred_fake, True)
        else:
            fake_AB = torch.cat((self.real_A, self.fake_B), 1)
            pred_fake = self.netD_Decompostion(fake_AB)
            self.loss_G_GAN_A = self.criterionGAN(pred_fake, True)
            fake_AB_B = torch.cat((self.real_B, self.fake_A), 1)
            pred_fake = self.netD_Generation(fake_AB_B)
            self.loss_G_GAN_B = self.criterionGAN(pred_fake, True)

            fake_AB = torch.cat((self.real_A, self.rec_A), 1)
            pred_fake = self.netD_Decompostion(fake_AB)
            self.loss_G_GAN_recA = self.criterionGAN(pred_fake, True)
            fake_AB_B = torch.cat((self.real_B, self.rec_B), 1)
            pred_fake = self.netD_Generation(fake_AB_B)
            self.loss_G_GAN_recB = self.criterionGAN(pred_fake, True)

        ## Flash L1 loss
        if self.opt.lambda_comp != 0:
            self.loss_G_L1_A_comp = self.criterionL1(self.flash_from_decomposition, self.real_C) * self.opt.lambda_comp
            self.loss_G_L1_B_comp = self.criterionL1(self.flash_from_generation, self.real_C) * self.opt.lambda_comp


        ## Flash Color Loss
        if self.opt.lambda_color_uv != 0:
            fake_C_A = kornia.rgb_to_yuv(self.flash_from_decomposition)[:,1:2,:,:]

            fake_C_B = kornia.rgb_to_yuv(self.flash_from_generation)[:,1:2,:,:]

            real_C_color = kornia.rgb_to_yuv(self.real_C)[:,1:2,:,:]

            self.loss_G_L1_A_comp_color = self.criterionL1(fake_C_A, real_C_color) * self.opt.lambda_color_uv
            self.loss_G_L1_B_comp_color = self.criterionL1(fake_C_B, real_C_color) * self.opt.lambda_color_uv


        if epoch >= self.opt.cycle_epoch:
            self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * self.opt.lambda_A
            self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * self.opt.lambda_B

        if self.opt.dslr_color_loss:
            real_A_blurred = kornia.gaussian_blur2d(self.real_A, (21, 21), (3, 3))
            real_B_blurred = kornia.gaussian_blur2d(self.real_B, (21, 21), (3, 3))
            fake_A_blurred = kornia.gaussian_blur2d(self.fake_A, (21, 21), (3, 3))
            fake_B_blurred = kornia.gaussian_blur2d(self.fake_B, (21, 21), (3, 3))
            self.loss_color_dslr_A = self.criterionL1(real_A_blurred, fake_A_blurred) * self.opt.dslr_color_loss
            self.loss_color_dslr_B = self.criterionL1(real_B_blurred, fake_B_blurred) * self.opt.dslr_color_loss


        self.loss_G_L1_A = self.criterionL1(self.fake_A, self.real_A) * self.opt.lambda_L1
        self.loss_G_L1_B = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1

        self.loss_G =self.loss_color_dslr_A + self.loss_color_dslr_B + \
                     self.loss_G_L1_B_comp_color + self.loss_G_L1_A_comp_color +\
                     self.loss_G_GAN_flash_A + self.loss_G_GAN_flash_B +\
                     self.loss_G_GAN_B + self.loss_G_GAN_A +\
                     self.loss_cycle_A + self.loss_cycle_B +\
                     self.loss_G_L1_A+ self.loss_G_L1_B +\
                     self.loss_G_L1_A_comp + self.loss_G_L1_B_comp

        self.loss_G.backward()

    def optimize_parameters(self, epoch):
        self.forward()  # compute fake images: G(A)
        # update D
        if self.opt.D_flash:
            self.set_requires_grad([self.netD_Flash], True)  # enable backprop for D
            self.optimizer_D1.zero_grad()  # set D's gradients to zero
            self.optimizer_D2.zero_grad()  # set D's gradients to zero
            self.backward_D_F_1()
            self.optimizer_D1.step()
            self.backward_D_F_2()
            self.optimizer_D2.step()

            # update G
            self.set_requires_grad([self.netD_Flash], False)  # D requires no gradients when optimizing G
            self.optimizer_G.zero_grad()  # set G's gradients to zero
            self.backward_G(epoch)  # calculate graidents for G
            self.optimizer_G.step()
        else:
            self.set_requires_grad([self.netD_Decompostion, self.netD_Generation], True)  # enable backprop for D
            self.optimizer_D.zero_grad()  # set D's gradients to zero
            self.backward_D_A()      # calculate gradients for D_A
            self.backward_D_B()
            self.optimizer_D.step()  # update D's weights
            # update G
            self.set_requires_grad([self.netD_Decompostion, self.netD_Generation], False)  # D requires no gradients when optimizing G
            self.optimizer_G.zero_grad()  # set G's gradients to zero
            self.backward_G(epoch)  # calculate graidents for G
            self.optimizer_G.step()  # udpate G's weights


def showImage(img,title=None):
    image = (img + 1)/2
    image = image.clone().detach().cpu().numpy().squeeze()
    image = np.transpose(image,[1,2,0])
    plt.imshow(image, cmap= 'inferno')
    plt.colorbar()
    if title is not None:
        plt.title(title)
    plt.show()

