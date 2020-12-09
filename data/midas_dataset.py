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
# import cv2
import argparse
from skimage.color import xyz2rgb
from skimage.color import xyz2lab
from skimage.color import lab2rgb
from skimage.color import rgb2xyz
from PIL.PngImagePlugin import PngImageFile, PngInfo
import random
import torchvision.transforms as transforms


class MidasDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
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
        # self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
        # self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        real_index = index % self.A_size
        A_path = self.A_paths[real_index]  # make sure index is within then range
        # print(index)
        # if self.opt.serial_batches:   # make sure index is within then range
        #     index_B = index % self.B_size
        # else:   # randomize the index for domain B to avoid fixed pairs.
        #     index_B = random.randint(0, self.B_size - 1)
        # print("b index {}".format(index_B))
        B_path_A = self.B_paths_A[real_index]
        B_path_B = self.B_paths_B[real_index]
        AB = Image.open(A_path).convert('RGB')
        midas_A = Image.open(B_path_A)
        midas_B = Image.open(B_path_B)
        midas_A = midas_A.resize((self.opt.crop_size,self.opt.crop_size))
        midas_B = midas_B.resize((self.opt.crop_size, self.opt.crop_size))
        # midas = skimage.img_as_float(midas)
        # midas = np.reshape(midas, [self.opt.crop_size, self.opt.crop_size,1])
        # midas = np.concatenate((midas, midas, midas), axis=2)
        # midas = (midas * 255 / np.max(midas)).astype('uint8')
        # print(midas.shape)
        # midas = Image.fromarray(midas)
        # print(AB.size)

        targetImage = PngImageFile(A_path)
        des = int(targetImage.text['des'])
        # print("index")
        # print(index)

        # des = int(AB.info['des'])
        # matrix = im.info['Comment']
        # split AB image into A and B
        w, h = AB.size
        w2 = int(w / 2)
        A = AB.crop((0, 0, w2, h))
        B = AB.crop((w2, 0, w, h))
        # A = A.resize((256,256))
        # B = B.resize((256, 256))
        flash = skimage.img_as_float(B)
        ambient = skimage.img_as_float(A)
        flash = makeBright(flash, 1.2)
        ambient = makeBright(ambient, 0.4)
        if des == 21:
            flash = changeTemp(flash,48,des)
        # A = ambient * self.opt.ambient_A + flash * self.opt.flash_A
        # B = ambient * self.opt.ambient_B + flash * self.opt.flash_B
        A= flash * 0.8 + ambient * 1.4
        B = ambient * 1.4
        C = A - B
        A = xyztorgb(A,des)
        B = xyztorgb(B, des)
        C = xyztorgb(C,des)
        # B = flash * 2.2
        transform_params = get_params(self.opt, A.size)
        A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))
        midas_transform = get_transform(self.opt, transform_params, grayscale=True)
        A = A_transform(A)
        B = B_transform(B)
        C = A_transform(C)
        midas_A = midas_transform(midas_A)
        midas_B = midas_transform(midas_B)
        # apply image transformation
        # print("a size {}".format(A.size()))
        # print("b size {}".format(B.size()))
        return {'A': A, 'B': B, 'C': C, 'midas_A': midas_B, 'midas_B': midas_A, 'A_random':A,'A_paths': A_path, 'B_paths_A': B_path_A, 'B_paths_B': B_path_B}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
def xyztorgb(image,des):
    illum = chromaticityAdaptation(des)
    height,width,c = image.shape
    # print(height)
    # print(width)
    out = np.matmul(image, illum)

    # for i in range(height):
    #     for j in range(width):
    #         xyz = image[i,j,:]
    #         X = xyz[0] * illum[0][0] + xyz[1]* illum[0][1] + xyz[2] * illum[0][2]
    #         Y = xyz[0] * illum[1][0] + xyz[1]* illum[1][1] + xyz[2] * illum[1][2]
    #         Z = xyz[0] * illum[2][0] + xyz[1]* illum[2][1] + xyz[2] * illum[2][2]
    #         # r =  3.2404542*X - 1.5371385*Y - 0.4985314*Z
    #         # g = -0.9692660*X + 1.8760108*Y + 0.0415560*Z
    #         # b =  0.0556434*X - 0.2040259*Y + 1.0572252*Z
    #         # if r<0:
    #         #     r = 0
    #         # if g< 0:
    #         #     g = 0
    #         # if b< 0:
    #         #     b = 0
    #         #
    #         # r = adj(r)
    #         # g = adj(g)
    #         # b = adj(b)
    #
    #         out[i,j,:] = [X,Y,Z]
    #
    out = xyz2rgb(out)
    out = (out * 255 / np.max(out)).astype('uint8')
    out = Image.fromarray(out)
    return out
def xyztolab(image,des):
    illum = chromaticityAdaptation(des)
    out = np.matmul(image, illum)

    out1 = xyz2lab(out).astype(np.float32)
    # print("lab")


    # out2 = (out1 * 255 / np.max(out1)).astype('uint8')

    # out = Image.fromarray(out1)
    return out1
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
        # print("0")
        # print(out[:,:,0])
        # print("1")
        # print(out[:, :, 1])
        # print("2")
        # print(out[:, :, 2])

    return out
