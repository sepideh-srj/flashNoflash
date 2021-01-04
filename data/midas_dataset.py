import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import os.path
import os
from PIL import ImageMath
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image
import skimage
import numpy as np
import cv2
import argparse
from skimage.color import xyz2rgb
from skimage.color import xyz2lab
from skimage.color import lab2rgb
from skimage.color import rgb2xyz
from PIL.PngImagePlugin import PngImageFile, PngInfo
import torchvision.transforms as transforms
from util import util
import torch

class MidasDataset(BaseDataset):

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot,opt.phase)  # create a path '/path/to/data/trainA'
        self.dir_B_A = os.path.join(opt.dataroot, opt.phase + '_midas')  # create a path '/path/to/data/trainB'
        self.dir_B_B = os.path.join(opt.dataroot, opt.phase + '_midas_A+B')  # create a path '/path/to/data/trainB'
        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.B_paths_A = sorted(make_dataset(self.dir_B_A, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        self.B_paths_B = sorted(make_dataset(self.dir_B_B, opt.max_dataset_size))  # load images from '/path/to/data/trainB'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths_A)  # get the size of dataset B
        btoA = self.opt.direction == 'BtoA'
        self.input_nc = self.opt.output_nc if btoA else self.opt.input_nc       # get the number of channels of input image
        self.output_nc = self.opt.input_nc if btoA else self.opt.output_nc      # get the number of channels of output image

    def __getitem__(self, index):

        real_index = index % self.A_size
        A_path = self.A_paths[real_index]  # make sure index is within then range
        AB = Image.open(A_path).convert('RGB')
        B_path_A = self.B_paths_A[real_index]
        B_path_B = self.B_paths_B[real_index]
        midas_A = Image.open(B_path_A)
        # midas_A.save("midas_B0.png")
        midas_B = Image.open(B_path_B)
        # midas_B.save("midas_A0.png")
        midas_A = np.asarray(midas_A)
        midas_A = midas_A/65535

        midas_A_normal = normal(midas_A)
        flash_pos_initial_A = (np.array(midas_A_normal.shape[0:2])/ 2).astype(int)
        flash_dir_initial_A = get_flash_dir(flash_pos_initial_A, midas_A_normal.shape)
        normal_dir_diff_initial_A = 1 + np.sum(np.multiply(midas_A_normal, flash_dir_initial_A), axis=2)
        normal_dir_diff_initial_A = np.multiply(normal_dir_diff_initial_A, 1 + midas_A / 3)

        midas_A = (midas_A * 255).astype('uint8')
        midas_A = Image.fromarray(midas_A)
        midas_A_normal = ((midas_A_normal+1) * 255 / (1+1)).astype('uint8')
        midas_A_normal = Image.fromarray(midas_A_normal)

        normal_dir_diff_initial_A = ((normal_dir_diff_initial_A) * 255 / (2.7)).astype('uint8')
        normal_dir_diff_initial_A = Image.fromarray(normal_dir_diff_initial_A)
        # normal_dir_diff_initial_A.save('midas_B.jpg')


        midas_B = np.asarray(midas_B)
        midas_B = midas_B / 65535
        midas_B_normal = normal(midas_B)

        flash_pos_initial_B = (np.array(midas_B_normal.shape[0:2])/ 2).astype(int)
        flash_dir_initial_B = get_flash_dir(flash_pos_initial_B, midas_B_normal.shape)
        normal_dir_diff_initial_B = 1 + np.sum(np.multiply(midas_B_normal, flash_dir_initial_B), axis=2)
        normal_dir_diff_initial_B = np.multiply(normal_dir_diff_initial_B, 1 + midas_B / 3)

        midas_B = (midas_B * 255).astype('uint8')
        midas_B = Image.fromarray(midas_B)
        # midas_B.save('midas_A.jpg')
        midas_B_normal = ((midas_B_normal + 1) * 255 / (1 + 1)).astype('uint8')
        midas_B_normal = Image.fromarray(midas_B_normal)

        normal_dir_diff_initial_B = ((normal_dir_diff_initial_B) * 255 / (2.7)).astype('uint8')
        normal_dir_diff_initial_B = Image.fromarray(normal_dir_diff_initial_B)
        # normal_dir_diff_initial_B.save('midas_A.jpg')

        targetImage = PngImageFile(A_path)
        des = int(targetImage.text['des'])
        w, h = AB.size
        w2 = int(w / 2)
        A = AB.crop((0, 0, w2, h))
        B = AB.crop((w2, 0, w, h))
        flash = skimage.img_as_float(B)
        ambient = skimage.img_as_float(A)
        flash = makeBright(flash, 1.2)
        if des == 21:
            flash = changeTemp(flash,48,des)
        # print('flash is', flash.shape)
        A = flash + ambient
        B = ambient

        B = xyztorgb(B, des)
        A = xyztorgb(A,des)
        transform_params = get_params(self.opt, A.size)
        A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))
        midas_transform = get_transform(self.opt, transform_params, grayscale=True)
        midas_normal_transform =get_transform(self.opt, transform_params)
        A = A_transform(A)
        B = B_transform(B)
        midas_A = midas_transform(midas_A)
        midas_B = midas_transform(midas_B)
        midas_A_normal = midas_normal_transform(midas_A_normal)
        midas_B_normal = midas_normal_transform(midas_B_normal)
        normal_dir_diff_initial_A = midas_transform(normal_dir_diff_initial_A)
        normal_dir_diff_initial_B = midas_transform(normal_dir_diff_initial_B)
        # print(midas_A.shape)
        # print(torch.max(midas_A))
        # print(torch.min(midas_A))

        return {'A': A, 'B': B, 'normal_dir_diff_initial_A': normal_dir_diff_initial_B, 'normal_dir_diff_initial_B': normal_dir_diff_initial_A, 'midas_A_normal': midas_B_normal,  'midas_B_normal': midas_A_normal, 'midas_A': midas_B, 'midas_B': midas_A, 'A_random':A,'A_paths': A_path, 'B_paths_A': B_path_A, 'B_paths_B': B_path_B}
    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.A_paths)

def xyztorgb(image,des):
    illum = chromaticityAdaptation(des)

    mat = [[3.2404542, -0.9692660, 0.0556434], [-1.5371385, 1.8760108, -0.2040259], [ -0.4985314,  0.0415560, 1.0572252]]
    image = np.matmul(image, illum)
    image = np.matmul(image, mat)
    image = np.where(image < 0, 0, image)
    image = np.where(image > 1, 1, image)
    out = (image * 255).astype('uint8')
    out = Image.fromarray(out)
    return out
def makeBright(image,ratio):
    image[:,:,0] = image[:,:,0]*ratio
    image[:,:,1] = image[:,:,1]*ratio
    image[:,:,2]= image[:,:,2]*ratio
    return image
def chromaticityAdaptation(calibrationIlluminant):
    if (calibrationIlluminant == 17):
        illum = [[0.8652435 , 0.0000000,  0.0000000],
             [0.0000000,  1.0000000,  0.0000000],
             [0.0000000,  0.0000000,  3.0598005]]
    elif (calibrationIlluminant == 19):
        illum = [[0.9691356,  0.0000000,  0.0000000],
              [0.0000000,  1.0000000,  0.0000000],
              [0.0000000,  0.0000000,  0.9209267]]
    elif (calibrationIlluminant == 20):
        illum =[[0.9933634,  0.0000000,  0.0000000],
               [0.0000000,  1.0000000, 0.0000000],
               [0.0000000,  0.0000000  ,1.1815972]]
    elif (calibrationIlluminant == 21):
        illum = [[1, 0,  0],
                    [0,  1,  0],
                    [0,  0,  1]]
    elif (calibrationIlluminant == 23):
      illum = [[1.0077340,  0.0000000,  0.0000000],
              [0.0000000,  1.0000000,  0.0000000],
              [0.0000000,  0.0000000,  0.8955170]]
    return illum
def normal(depth):
    g_x = cv2.Sobel(depth, cv2.CV_64F, 0, 1, ksize=5)
    g_y = cv2.Sobel(depth, cv2.CV_64F, 1, 0, ksize=5)
    g_z = np.ones_like(depth)
    normal = np.dstack((-g_x, -g_y, g_z))
    n = np.linalg.norm(normal, axis=2)
    normal[:, :, 0] /= n
    normal[:, :, 1] /= n
    normal[:, :, 2] /= n
    # print('normal is', normal)
    return normal

def get_flash_dir(flash_pos,shape):
    y, x = np.meshgrid(range(0, shape[1], 1),
                       range(0, shape[0], 1))
    flash_dir = np.zeros((shape[0],shape[1],3))
    flash_dir[:, :, 0] = -x + flash_pos[0]
    flash_dir[:, :, 1] = -y + flash_pos[1]
    flash_dir[:, :, 2] = max(shape)
    n_dir = np.linalg.norm(flash_dir, axis=2)
    flash_dir[:, :, 0] /= n_dir
    flash_dir[:, :, 1] /= n_dir
    flash_dir[:, :, 2] /= n_dir
    return flash_dir

def getRatio(t, low, high):
    dist = t - low
    range = (high-low)/100
    return dist/range
def changeTemp(image, tempChange,des):

    if (des == 17):
        t1 = 5500
        if tempChange == 44:
            tempChange = -400
        elif tempChange == 40:
            tempChange = - 450
        elif tempChange == 52:
            tempChange = 234
        elif tempChange == 54:
            tempChange = 468
        t = t1 + tempChange
        # print(t)
        if t <= 10000 and t> 6500:
            # print("here3")
            r=getRatio(t, 6500, 10000)
            xD = 0.3
            yD = 0.3
            xS = 0.3118
            yS = 0.3224
            r_x=(xD-xS) / 100
            xD=xS+r * r_x
            r_y=(yD-yS) / 100
            yD=yS+r * r_y

        elif 6500 >= t and t > 5500:
            # print("here2")
            r=getRatio(t, 5500, 6500)
            xD = 0.3118
            yD = 0.3224
            xS = 0.3580
            yS = 0.3239
            r_x=(xD-xS) / 100
            xD=xS+r * r_x
            r_y=(yD-yS) / 100
            yD=yS+r * r_y

        elif 5000 <= t and t < 5500:
            # print("here")
            r=getRatio(t, 5500, 5000)
            xD = 0.3752
            yD = 0.3238
            xS = 0.3580
            yS = 0.3239
            r_x=(xD-xS) / 100
            xD=xS+r * r_x
            r_y=(yD-yS) / 100
            yD=yS+r * r_y

        elif (t >= 4500 and t < 5000):
            # print("here6")
            r=getRatio(t, 5000, 4500)

            xD = 0.4231
            yD = 0.3304
            xS = 0.3752
            yS = 0.3238
            r_x=(xD-xS) / 100
            xD=xS+r * r_x
            r_y=(yD-yS) / 100
            yD=yS+r * r_y

        elif t < 4500 and t>= 4000:
            # print("here7")
            r=getRatio(t, 4500, 4000)
            xD = 0.4949
            yD = 0.3564
            xS = 0.4231
            yS = 0.3304
            r_x=(xD-xS) / 100
            xD=xS+r * r_x
            r_y=(yD-yS) / 100
            yD=yS+r * r_y

        elif t >= 0 and t < 4000:
            # print("here9")
            r=getRatio(t, 4000, 0)
            xD =  0.5041
            yD =  0.3334
            xS = 0.4949
            yS = 0.3564
            r_x=(xD-xS) / 100
            xD=xS+r * r_x
            r_y=(yD-yS) / 100
            yD=yS+r * r_y

        chromaticity_x = 0.3580
        chromaticity_y = 0.3239
    elif des == 21:
        t1 = 4500
        if tempChange == 48:
            tempChange = 700
        if tempChange == 44:
            tempChange = 400
        elif tempChange == 40:
            tempChange = 250
        elif tempChange == 52:
            tempChange = 800
        elif tempChange == 54:
            tempChange = 1000

        t = tempChange + t1
        if t >= 4500 and t <= 7500:
            # print("desss")
            r=getRatio(t, 4500, 7500)
            xD = 0.17
            yD = 0.17
            xS = 0.4231
            yS = 0.3304
            r_x=(xD-xS) / 100
            xD=xS+r * r_x
            r_y=(yD-yS) / 100
            yD=yS+r * r_y

        elif (t < 4500 and t>= 4000):

            r=getRatio(t, 4500, 4000)
            xD = 0.4949
            yD = 0.3564
            xS = 0.4231
            yS = 0.3304
            r_x=(xD-xS) / 100
            xD=xS+r * r_x
            r_y=(yD-yS) / 100
            yD=yS+r * r_y

        elif (t >= 3500 and t< 4000):

            r=getRatio(t, 4000, 3500)
            xD =  0.5141
            yD =  0.3434
            xS = 0.4949
            yS = 0.3564
            r_x=(xD-xS) / 100
            xD=xS+r * r_x
            r_y=(yD-yS) / 100
            yD=yS+r * r_y
        elif (t > 0 and t < 3500):
            r=getRatio(t, 3500, 0)
            xD = 0.5189
            yD = 0.3063
            xS =  0.5141
            yS =  0.3434
            r_x=(xD-xS) / 100
            xD=xS+r * r_x
            r_y=(yD-yS) / 100
            yD=yS+r * r_y

        chromaticity_x = 0.4231

        chromaticity_y = 0.3304

    offset_x = xD/chromaticity_x
    offset_y = yD / chromaticity_y
    # image_numpy = np.where(image_numpy > 255, 255, image_numpy)
    out = image
    h, w, c = image.shape
    img0 = image[:, :, 0]
    img1 = image[:, :, 1]
    img2 = image[:, :, 2]
    sumImage = img0 + img1 + img2
    x_pix = np.zeros((h, w))
    y_pix = np.zeros((h, w))

    nonZeroSum = np.where(sumImage != 0)
    x_pix[nonZeroSum] = img0[nonZeroSum] / sumImage[nonZeroSum]
    y_pix[nonZeroSum] = img1[nonZeroSum] / sumImage[nonZeroSum]

    x_pix = x_pix * offset_x
    y_pix = y_pix * offset_y

    out0 = np.zeros((h, w))
    out2 = np.zeros((h, w))

    nonZeroY = np.where(y_pix != 0)
    ones = np.ones((h, w))
    out0[nonZeroY] = x_pix[nonZeroY] * img1[nonZeroY] / y_pix[nonZeroY]
    out2[nonZeroY] = (ones[nonZeroY] - x_pix[nonZeroY] - y_pix[nonZeroY]) * img1[nonZeroY] / y_pix[nonZeroY]
    out[:,:,0] = out0
    out[:,:,2] = out2

    return out