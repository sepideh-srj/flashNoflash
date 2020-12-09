import os.path
import os

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
from skimage.color import xyz2rgb
from PIL.PngImagePlugin import PngImageFile, PngInfo
import random
import torchvision.transforms as transforms

class AlignedLabDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        self.AB_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size))  # get image paths
        assert (self.opt.load_size >= self.opt.crop_size)  # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        AB_path = self.AB_paths[index]
        # print(AB_path)
        path = AB_path.replace('train', 'train2')
        # print(index)
        AB = Image.open(AB_path)
        targetImage = PngImageFile(AB_path)
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
        flash = skimage.img_as_float(B)
        ambient = skimage.img_as_float(A)
        flash = makeBright(flash,1.2)
        # ambient = makeBright(ambient؟,0.5)
        if des == 21:
            flash = changeTemp(flash,48,des)
        # if des == 21:
        #     flash = changeTemp(flash,48,des)
        # im = A_float * opacity / 100 + B_float * (100 - opacity) / 100
        # paper version 4: from A flash 0.5 ambient 1.7 to flash 1.7 ambient 0.5
        # A = flash * 0.6+ ambient * 1.6
        # A = xyztorgb(A,des)
        # # opacity2 = opacity + 0.7
        # # if opacity2 > 2:
        # #     opacity2 = 2
        # B = flash * 1.1 + ambient * 1.1
        # B = xyztorgb(B,des)
        # flash = changeTemp(flash,44,des)
        # C = flash * 0.4+ ambient * 1.8
        # C = xyztorgb(C,des)
        # opacity2 = opacity + 0.7
        # if opacity2 > 2:
        #     opacity2 = 2
        # B = flash * 1.1 + ambient * 1.1
        # B = xyztorgb(B,des)
        # if des == 17:
        #     print("here2")
        #     A = flash* 1.8 + ambient * 0.4
        #     A = xyztorgb(A,des)
        #     flash2 = changeTemp(flash, 52, des)
        #     B = flash2 * 1.8 + ambient *0.4
        #     B = xyztorgb(B,des)
        #     flash3 = changeTemp(flash, 54, des)
        #     C = flash3 * 1.8 + ambient * 0.4
        #     C = xyztorgb(C, des)
        # else:
        #     print("here")
        #     flash1 = changeTemp(flash, 54, des)
        #     A = flash1* 1.8 + ambient * 0.4
        #     A = xyztorgb(A,des)
        #     flash2 = changeTemp(flash, 52, des)
        #     B = flash2 * 1.8 + ambient *0.4
        #     B = xyztorgb(B, des)
        #     flash3 = changeTemp(flash, 40, des)
        #     C = flash3 * 1.8 + ambient *0.4
        #     C = xyztorgb(C, des)
        # C = flash * 0.4 + ambient * 1.8

        if self.opt.random:
            r = random.uniform(0.7, 1.6)
            A = flash.copy() * 0.8 + ambient * r
            B = ambient * r
        else:
            A= flash + ambient
            B = ambient

        if self.opt.lab:
            A = xyztolab(A, des)
            B = xyztolab(B, des)
            C = xyztolab(C, des)
        else:
            A = xyztorgb(A, des)
            B = xyztorgb(B, des)
            C = A
        # cv2.imwrite(path_AB, im_AB)
        # im = (im * 255 / np.max(im)).astype('uint8')

        # print(blended)
        # im = (im * 255 / np.max(im)).astype('uint8')
        # im = Image.fromarray(im)

        transform_params = get_params(self.opt, A.size)
        # apply the same transform to both A and B
        A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))
        # A = transforms.ToTensor()(A)
        # B = transforms.ToTensor()(B)
        # C = transforms.ToTensor()(C)
        A = A_transform(A)
        B = B_transform(B)
        C = B_transform(C)

        return {'A': A, 'B': B, 'C': C, 'A_paths': AB_path, 'B_paths': AB_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)

def xyztorgb(image,des):
    illum = chromaticityAdaptation(des)
    height,width,c = image.shape
    # print(height)
    # print(width)
    # out = np.matmul(image, illum)
    mat = [[3.2404542, -0.9692660, 0.0556434], [-1.5371385, 1.8760108, -0.2040259], [ -0.4985314,  0.0415560, 1.0572252]]
    image = np.matmul(image, illum)
    image = np.matmul(image, mat)
    image = np.where(image < 0, 0, image)
    image = np.where(image > 1, 1, image)
    out = (image * 255).astype('uint8')
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
# def  adj(C):
#     if (C >0 and C < 0.0031308):
#         return 12.92 * C
#
#     return 1.055 * (C**0.41666) - 0.055
#


# def getXYZ(imgFloat, colorMatrix, calibrationIlluminant, size):
#     # imgFloat = img_as_float(image)
#     XYZtoCamera = np.reshape(colorMatrix, (3, 3), order='F')
#     XYZtoCamera = np.transpose(XYZtoCamera)
#     width, height = size
#
#     imf = np.reshape(imgFloat, [width * height, 3], order='F')
#     imf = np.transpose(imf)
#
#     XYZtoCamera = np.linalg.inv(XYZtoCamera)
#     imf = np.dot(XYZtoCamera,imf);
#     imf = np.transpose(imf)
#     imf = np.reshape(imf, [height, width, 3], order='F')
#     # zarib = fixWhitePoint(calibrationIlluminant)
#     # imf[:, :, 0] = zarib[0] * imf[:, :, 0]
#     # imf[:, :, 2] = zarib[2] * imf[:, :, 2]
#     return imf


