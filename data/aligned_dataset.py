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
from PIL.PngImagePlugin import PngImageFile, PngInfo
import random


class AlignedDataset(BaseDataset):
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


        # des = int(AB.info['des'])
        # matrix = im.info['Comment']
        # split AB image into A and B
        w, h = AB.size
        w2 = int(w / 2)
        A = AB.crop((0, 0, w2, h))
        B = AB.crop((w2, 0, w, h))
        opacity = 50
        flash = skimage.img_as_float(A)
        ambient = skimage.img_as_float(B)
        # im = A_float * opacity / 100 + B_float * (100 - opacity) / 100
        # paper version 4: from A flash 0.5 ambient 1.7 to flash 1.7 ambient 0.5
        A = flash * 1.1+ ambient * 1.1
        A = xyztorgb(A,des)
        # opacity2 = opacity + 0.7
        # if opacity2 > 2:
        #     opacity2 = 2
        B = flash * 2.2 + ambient * 1.1
        B = xyztorgb(B,des)

        # cv2.imwrite(path_AB, im_AB)
        # im = (im * 255 / np.max(im)).astype('uint8')

        # print(blended)
        # im = (im * 255 / np.max(im)).astype('uint8')
        # im = Image.fromarray(im)

        # apply the same transform to both A and B
        transform_params = get_params(self.opt, A.size)
        A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))

        A = A_transform(A)
        B = B_transform(B)

        return {'A': A, 'B': B, 'A_paths': AB_path, 'B_paths': AB_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)
def xyztorgb(image,des):
    illum = chromaticityAdaptation(des)
    height,width,c = image.shape
    # print(height)
    # print(width)
    out = image
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
    out = (out * 255).astype('uint8')
    out = Image.fromarray(out)
    return out

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


