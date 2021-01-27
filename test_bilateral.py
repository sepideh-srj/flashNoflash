"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from --checkpoints_dir and save the results to --results_dir.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for --num_test images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
import torch
import numpy as np
import cv2
from cv2.ximgproc import jointBilateralFilter
# from cv2.ximgproc import guidedFilter

from guidedfilter import GuidedFilter

import matplotlib.pyplot as plt

def guidedFilter3(guide,source,sigmaSpace=16):
    result = np.zeros_like(source)
    result[:,:,0] =GuidedFilter(guide[:,:,0], source[:,:,0], sigmaSpace, 0.00001).smooth.astype('float32')
    result[:,:,1] =GuidedFilter(guide[:,:,1], source[:,:,1], sigmaSpace, 0.00001).smooth.astype('float32')
    result[:,:,2] =GuidedFilter(guide[:,:,2], source[:,:,2], sigmaSpace, 0.00001).smooth.astype('float32')
    return result

def showImage(img,title=None):
    img = (img+1)/2
    plt.imshow(img, cmap= 'inferno')
    plt.colorbar()
    if title is not None:
        plt.title(title)
    plt.show()

def gama_corect(rgb):
    srgb = np.zeros_like(rgb)
    mask1 = (rgb > 0) * (rgb < 0.0031308)
    mask2 = (1 - mask1).astype(bool)
    srgb[mask1] = 12.92 * rgb[mask1]
    srgb[mask2] = 1.055 * np.power(rgb[mask2], 0.41666) - 0.055
    srgb[srgb < 0] = 0
    srgb[srgb > 1] = 1
    return srgb

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers

    result_dir = 'results/' + opt.name
    A_dir = result_dir + '/' + '1-A_images'
    B_dir = result_dir + '/' + '4-B_images'
    A_fake_dir = result_dir + '/' + '2-A_fake_images'
    B_fake_dir = result_dir + '/' + '5-B_fake_images'
    A_fake_filtered_dir = result_dir + '/' + '3-A_fake_filtered_images'
    B_fake_filtered_dir = result_dir + '/' + '6-B_fake_filtered_images'
    A_ratio_dir = result_dir + '/' + '7-A_ratio_images'
    A_ratio_filtered_dir = result_dir + '/' + '8-A_ratio_filtered_images'
    B_ratio_dir = result_dir + '/' + '9-B_ratio_images'
    B_ratio_filtered_dir = result_dir + '/' + '10-B_filtered_ratio_images'


    dirs = [A_dir, B_dir, A_fake_dir, B_fake_dir, A_fake_filtered_dir,
            B_fake_filtered_dir,A_ratio_dir,A_ratio_filtered_dir,
            B_ratio_dir,B_ratio_filtered_dir]

    for dir in dirs:
        if not os.path.exists(dir):
            os.makedirs(dir)

    if opt.eval:
        model.eval()
    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break

        img_path = data['A_paths'][0]
        img_path = img_path.replace(opt.dataroot, '')
        seprator = '_'
        img_path = seprator.join(img_path.split('/'))
        print('processing (%04d)-th image... %s' % (i, img_path))


        model.set_input(data)  # unpack data from data loader
        with torch.no_grad():
            model.forward_bilateral()           # run inference

        A = data['A_org'].cpu().squeeze().numpy().astype('float32')
        B = data['B_org'].cpu().squeeze().numpy().astype('float32')
        A = 2*A - 1
        B = 2*B - 1

        A_ratio = model.fake_A.cpu().squeeze().numpy().astype('float32')
        B_ratio = model.fake_B.cpu().squeeze().numpy().astype('float32')

        A_ratio = np.transpose(A_ratio, (1, 2, 0))
        B_ratio = np.transpose(B_ratio, (1, 2, 0))

        A_ratio = cv2.resize(A_ratio,(A.shape[1],A.shape[0]))
        B_ratio = cv2.resize(B_ratio,(B.shape[1],B.shape[0]))

        #TODO: add bilateral filter

        # A_ratio_filtered = jointBilateralFilter(B, A_ratio, d=0, sigmaColor=0.001, sigmaSpace=10)
        # B_ratio_filtered = jointBilateralFilter(A, B_ratio, d=0, sigmaColor=0.001, sigmaSpace=10)
        ratio = int(min(A_ratio.shape[0:2])/50)

        A_ratio_filtered = guidedFilter3(B,A_ratio,ratio)
        B_ratio_filtered = guidedFilter3(A,B_ratio,ratio)
        ##
        # showImage(A_ratio,'A-{}'.format(ratio))
        # showImage(A_ratio_filtered,'F-{}'.format(ratio))

        A_fake = (2 * (B * A_ratio + 2 * A_ratio + B) - 1) / 5
        B_fake = (2 * (A * B_ratio + 2 * B_ratio + A) - 1) / 5

        A_fake_filtered = (2 * (B * A_ratio_filtered + 2 * A_ratio_filtered + B) - 1) / 5
        B_fake_filtered = (2 * (A * B_ratio_filtered + 2 * B_ratio_filtered + A) - 1) / 5

        A_fake = (A_fake+1)/2
        B_fake = (B_fake+1)/2
        A_fake_filtered = (A_fake_filtered+1)/2
        B_fake_filtered = (B_fake_filtered+1)/2
        A = (A + 1) / 2
        B = (B + 1) / 2
        A_ratio = (A_ratio +1)/2
        B_ratio = (B_ratio +1)/2
        A_ratio_filtered = (A_ratio_filtered +1)/2
        B_ratio_filtered = (B_ratio_filtered +1)/2

        A = gama_corect(A)
        B = gama_corect(B)
        A_fake = gama_corect(A_fake)
        B_fake = gama_corect(B_fake)
        A_fake_filtered = gama_corect(A_fake_filtered)
        B_fake_filtered = gama_corect(B_fake_filtered)

        A = (A * 255).astype('uint8')
        B = (B * 255).astype('uint8')
        A_fake = (A_fake * 255).astype('uint8')
        B_fake = (B_fake * 255).astype('uint8')
        A_fake_filtered = (A_fake_filtered * 255).astype('uint8')
        B_fake_filtered = (B_fake_filtered * 255).astype('uint8')
        A_ratio = (A_ratio * 255).astype('uint8')
        B_ratio = (B_ratio * 255).astype('uint8')
        A_ratio_filtered = (A_ratio_filtered * 255).astype('uint8')
        B_ratio_filtered = (B_ratio_filtered * 255).astype('uint8')


        A = cv2.cvtColor(A, cv2.COLOR_RGB2BGR)
        B = cv2.cvtColor(B, cv2.COLOR_RGB2BGR)
        A_fake = cv2.cvtColor(A_fake, cv2.COLOR_RGB2BGR)
        B_fake = cv2.cvtColor(B_fake, cv2.COLOR_RGB2BGR)
        A_fake_filtered = cv2.cvtColor(A_fake_filtered, cv2.COLOR_RGB2BGR)
        B_fake_filtered = cv2.cvtColor(B_fake_filtered, cv2.COLOR_RGB2BGR)
        A_ratio = cv2.cvtColor(A_ratio, cv2.COLOR_RGB2BGR)
        B_ratio = cv2.cvtColor(B_ratio, cv2.COLOR_RGB2BGR)
        A_ratio_filtered = cv2.cvtColor(A_ratio_filtered, cv2.COLOR_RGB2BGR)
        B_ratio_filtered = cv2.cvtColor(B_ratio_filtered, cv2.COLOR_RGB2BGR)

        cv2.imwrite(A_dir + '/' + img_path[1:],A)
        cv2.imwrite(B_dir + '/' + img_path[1:],B)
        cv2.imwrite(A_fake_dir + '/' + img_path[1:],A_fake)
        cv2.imwrite(B_fake_dir + '/' + img_path[1:],B_fake)
        cv2.imwrite(A_fake_filtered_dir + '/' + img_path[1:],A_fake_filtered)
        cv2.imwrite(B_fake_filtered_dir + '/' + img_path[1:],B_fake_filtered)
        cv2.imwrite(A_ratio_dir + '/' + img_path[1:],A_ratio)
        cv2.imwrite(B_ratio_dir + '/' + img_path[1:],B_ratio)
        cv2.imwrite(A_ratio_filtered_dir + '/' + img_path[1:],A_ratio_filtered)
        cv2.imwrite(B_ratio_filtered_dir + '/' + img_path[1:],B_ratio_filtered)


