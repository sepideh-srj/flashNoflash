import os.path
import os

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
from skimage.color import xyz2rgb
from PIL.PngImagePlugin import PngImageFile, PngInfo
import random
import torchvision.transforms as transforms
import torch
import copy

#MIDAS
import midas.utils
from midas.models.midas_net import MidasNet
from midas.models.transforms import Resize, NormalizeImage, PrepareForNet
from torchvision.transforms import Compose
from depthmerge.options.test_options import TestOptions
from depthmerge.models.pix2pix4depth_model import Pix2Pix4DepthModel


import matplotlib.pyplot as plt
def showImage(img,title=None):
    plt.imshow(img, cmap= 'inferno')
    plt.colorbar()
    if title is not None:
        plt.title(title)
    plt.show()

class AlignedLabDataset(BaseDataset):

    def __init__(self, opt):

        BaseDataset.__init__(self, opt)
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        self.AB_paths = sorted(
            make_dataset(self.dir_AB, opt.max_dataset_size))  # load images from '/path/to/data/trainB'

        assert (self.opt.load_size >= self.opt.crop_size)  # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

        opt_merge = copy.deepcopy(opt)
        opt_merge.isTrain = False
        opt_merge.model = 'pix2pix4depth'
        self.mergenet = Pix2Pix4DepthModel(opt_merge)
        self.mergenet.save_dir = 'depthmerge/checkpoints/scaled_04_1024'
        self.mergenet.load_networks('latest')
        self.mergenet.eval()

        self.device = self.mergenet.device

        midas_model_path = "midas/model-f46da743.pt"
        self.midasmodel = MidasNet(midas_model_path, non_negative=True)
        self.midasmodel.to(self.device)
        self.midasmodel.eval()

        torch.multiprocessing.set_start_method('spawn')

    def gama_corect(self,rgb):
        srgb = np.zeros_like(rgb)
        mask1 = (rgb > 0) * (rgb < 0.0031308)
        mask2 = (1 - mask1).astype(bool)
        srgb[mask1] = 12.92 * rgb[mask1]
        srgb[mask2] = 1.055 * np.power(rgb[mask2], 0.41666) - 0.055
        srgb[srgb < 0] = 0
        return srgb

    def doubleestimate(self,img, size1, size2):
        estimate1 = self.singleestimate(img, size1)
        estimate1 = cv2.resize(estimate1, (1024, 1024), interpolation=cv2.INTER_CUBIC)

        estimate2 = self.singleestimate(img, size2)
        estimate2 = cv2.resize(estimate2, (1024, 1024), interpolation=cv2.INTER_CUBIC)

        self.mergenet.set_input(estimate1, estimate2)
        self.mergenet.test()
        visuals = self.mergenet.get_current_visuals()
        prediction_mapped = visuals['fake_B']
        prediction_mapped = (prediction_mapped + 1) / 2
        prediction_mapped = (prediction_mapped - torch.min(prediction_mapped)) / (
                torch.max(prediction_mapped) - torch.min(prediction_mapped))
        prediction_mapped = prediction_mapped.squeeze().cpu().numpy()

        prediction_end_res = cv2.resize(prediction_mapped, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)

        return prediction_end_res

    def singleestimate(self,img, msize):
        return self.estimateMidas(img, msize)

    def estimateMidas(self,img, msize):
        transform = Compose(
            [
                Resize(
                    msize,
                    msize,
                    resize_target=None,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=32,
                    resize_method="upper_bound",
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                PrepareForNet(),
            ]
        )

        img_input = transform({"image": img})["image"]
        # compute
        with torch.no_grad():
            sample = torch.from_numpy(img_input).to(self.device).unsqueeze(0)
            prediction = self.midasmodel.forward(sample)

        prediction = prediction.squeeze().cpu().numpy()
        prediction = cv2.resize(prediction, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)

        depth_min = prediction.min()
        depth_max = prediction.max()

        if depth_max - depth_min > np.finfo("float").eps:
            prediction = (prediction - depth_min) / (depth_max - depth_min)
        else:
            prediction = 0

        return prediction

    def estimateDepth(self,rgb_mix):
        # rgb_mix = (rgb_mix + 1) / 2
        # rgb_mix = rgb_mix.cpu().numpy()
        # rgb_mix = np.transpose(rgb_mix, (1, 2, 0))
        rgb_mix = self.gama_corect(rgb_mix)
        # print(rgb_mix.shape)
        # showImage(rgb_mix)
        depth_temp = self.doubleestimate(rgb_mix, 256, 512)
        # depth_temp = self.doubleestimate(rgb_mix, 384, 768)
        # showImage(depth_temp)
        return depth_temp


    def __getitem__(self, index):
        # read a image given a random integer index
        AB_path = self.AB_paths[index]
        # print(AB_path)

        # print(index)
        if 'flash' in AB_path.split('/')[-1]:
            roomDir = 'Rooms'

            roomDir = os.path.join(self.opt.dataroot, roomDir)
            # print(roomDir)
            # print("here")
            roomImages = os.listdir(roomDir)
            flash_color_adjustment_ratio = [1.23, 0.8, 1.04]
            room_image = roomImages[index % (len(roomImages) - 1)]
            path_room = os.path.join(roomDir, room_image)
            room = Image.open(path_room)
            targetImage = PngImageFile(path_room)
            des = int(targetImage.text['des'])
            w, h = room.size
            w2 = int(w / 2)
            A = room.crop((0, 0, w2, h))
            B = room.crop((w2, 0, w, h))
            A = A.crop((0, 0, 256, 256))
            B = B.crop((0, 0, 256, 256))

            flash = skimage.img_as_float(B)
            ambient = skimage.img_as_float(A)

            flash = makeBright(flash, 1.5)
            #
            # flash_avg = getAverage(flash[:, :, 0]) + getAverage(flash[:, :, 1]) + getAverage(flash[:, :, 2])
            # ambient_avg = getAverage(ambient[:, :, 0]) + getAverage(ambient[:, :, 1]) + getAverage(ambient[:, :, 2])
            # flash = flash * 2 * ambient_avg / (flash_avg + 0.0001)

            if des == 21:
                flash = changeTemp(flash, 48, des)
                ambient = changeTemp(ambient, 48, des)
            flash = flash + ambient
            flash = xyztorgb(flash, des)
            ambient = xyztorgb(ambient, des)

            # path_people = os.path.join(peopleDir, path)
            people_pic = Image.open(AB_path)
            w, h = people_pic.size
            w2 = int(w / 2)
            ambient_pic = people_pic.crop((0, 0, w2, h))
            flash_pic = people_pic.crop((w2, 0, w, h))

            flash_pic = flash_pic.resize((256, 256))
            ambient_pic = ambient_pic.resize((256, 256))
            flash_color_adjustment_ratio = flash_color_adjustment_ratio / np.max(flash_color_adjustment_ratio)
            flash_pic = lin(skimage.img_as_float(flash_pic))
            flash_pic[:, :, 0] = flash_pic[:, :, 0] * flash_color_adjustment_ratio[0]
            flash_pic[:, :, 1] = flash_pic[:, :, 1] * flash_color_adjustment_ratio[1]
            flash_pic[:, :, 2] = flash_pic[:, :, 2] * flash_color_adjustment_ratio[2]
            ambient_pic_lin = ambient_pic.copy()
            ambient_pic_lin = lin(skimage.img_as_float(ambient_pic_lin))
            flash_lin = lin(skimage.img_as_float(flash.copy()))
            ambient_lin = lin(skimage.img_as_float(ambient.copy()))

            avg_ambient_red_bg = getAverage(ambient_lin[:, :, 0])
            avg_ambient_green_bg = getAverage(ambient_lin[:, :, 1])
            avg_ambient_blue_bg = getAverage(ambient_lin[:, :, 2])

            avg_flash_red_bg = getAverage(flash_lin[:, :, 0])
            avg_flash_green_bg = getAverage(flash_lin[:, :, 1])
            avg_flash_blue_bg = getAverage(flash_lin[:, :, 2])

            bg_ratio_red = avg_ambient_red_bg / avg_flash_red_bg
            bg_ratio_green = avg_ambient_green_bg / avg_flash_green_bg
            bg_ratio_blue = avg_ambient_blue_bg / avg_flash_blue_bg

            ambient_pic_lin[:, :, 0] = ambient_pic_lin[:, :, 0] * bg_ratio_red
            ambient_pic_lin[:, :, 1] = ambient_pic_lin[:, :, 1] * bg_ratio_green
            ambient_pic_lin[:, :, 2] = ambient_pic_lin[:, :, 2] * bg_ratio_blue

            # ambient_pic_adjust = gama_corect(ambient_pic_lin)
            ambient_pic_adjust = Image.fromarray((ambient_pic_lin * 255).astype('uint8'))

            # flash_pic = gama_corect(flash_pic)
            flash_pic = Image.fromarray((flash_pic * 255).astype('uint8'))

            flash_out = alpha_blend(flash_pic, flash)
            ambient_out_adjust = alpha_blend(ambient_pic_adjust, ambient)

            flash_out = (flash_out * 255).astype('uint8')
            flash_out = Image.fromarray(flash_out)
            ambient_out_adjust = (ambient_out_adjust * 255).astype('uint8')
            ambient_out_adjust = Image.fromarray(ambient_out_adjust)

            A = flash_out
            B = ambient_out_adjust
        elif 'multi' in AB_path.split('/')[-1]:
            AB = Image.open(AB_path)
            w, h = AB.size
            w2 = int(w / 2)
            A = AB.crop((0, 0, w2, h))
            B = AB.crop((w2, 0, w, h))
            A = A.resize((256, 256))
            B = B.resize((256, 256))
            flash = lin(skimage.img_as_float(B))
            ambient = lin(skimage.img_as_float(A))



            rest_path = AB_path.replace('train','rest_multi').replace('.jpg','')+'_ambient.jpg'
            rest_Ambient = Image.open(rest_path)
            w, h = rest_Ambient.size
            w4 = int(w / 4)
            A2 = rest_Ambient.crop((0, 0, w4, h))
            A3 = rest_Ambient.crop((w4, 0, 2*w4, h))
            A4 = rest_Ambient.crop((2*w4, 0, 3*w4, h))
            A5 = rest_Ambient.crop((3*w4, 0, w, h))

            ambient2 = lin(skimage.img_as_float(A2))
            ambient3 = lin(skimage.img_as_float(A3))
            ambient4 = lin(skimage.img_as_float(A4))
            ambient5 = lin(skimage.img_as_float(A5))

            transform_params = get_params(self.opt, A.size)
            depth_transform = get_transform(self.opt, transform_params, grayscale=True)
            transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))

            depth_flash = Image.fromarray((self.estimateDepth(flash) * 255).astype('uint8'))
            depth_flash = depth_transform(depth_flash).unsqueeze(0)

            depth_ambient1 = Image.fromarray((self.estimateDepth(ambient) * 255).astype('uint8'))
            depth_ambient2 = Image.fromarray((self.estimateDepth(ambient2) * 255).astype('uint8'))
            depth_ambient3 = Image.fromarray((self.estimateDepth(ambient3) * 255).astype('uint8'))
            depth_ambient4 = Image.fromarray((self.estimateDepth(ambient4) * 255).astype('uint8'))
            depth_ambient5 = Image.fromarray((self.estimateDepth(ambient5) * 255).astype('uint8'))
            depth_ambient1 = depth_transform(depth_ambient1).unsqueeze(0)
            depth_ambient2 = depth_transform(depth_ambient2).unsqueeze(0)
            depth_ambient3 = depth_transform(depth_ambient3).unsqueeze(0)
            depth_ambient4 = depth_transform(depth_ambient4).unsqueeze(0)
            depth_ambient5 = depth_transform(depth_ambient5).unsqueeze(0)


            flash = (flash * 255).astype('uint8')
            ambient = (ambient * 255).astype('uint8')
            ambient2 = (ambient2 * 255).astype('uint8')
            ambient3 = (ambient3 * 255).astype('uint8')
            ambient4 = (ambient4 * 255).astype('uint8')
            ambient5 = (ambient5 * 255).astype('uint8')

            flash = Image.fromarray(flash)
            ambient = Image.fromarray(ambient)
            ambient2 = Image.fromarray(ambient2)
            ambient3 = Image.fromarray(ambient3)
            ambient4 = Image.fromarray(ambient4)
            ambient5 = Image.fromarray(ambient5)

            flash = transform(flash).unsqueeze(0)
            ambient = transform(ambient).unsqueeze(0)
            ambient2 = transform(ambient2).unsqueeze(0)
            ambient3 = transform(ambient3).unsqueeze(0)
            ambient4 = transform(ambient4).unsqueeze(0)
            ambient5 = transform(ambient5).unsqueeze(0)


            A_final = torch.cat((flash,flash,flash,flash,flash),dim=0)
            B_final = torch.cat((ambient,ambient2,ambient3,ambient4,ambient5),dim=0)

            depth_A_final = torch.cat((depth_flash,depth_flash,depth_flash,depth_flash,depth_flash),dim=0)
            depth_B_final = torch.cat((depth_ambient1,depth_ambient2,depth_ambient3,depth_ambient4,depth_ambient5),dim=0)

            return {'A': A_final, 'B': B_final , 'depth_A':depth_A_final , 'depth_B':depth_B_final, 'A_paths': AB_path, 'B_paths': AB_path}

        else:
            AB = Image.open(AB_path)
            targetImage = PngImageFile(AB_path)
            des = int(targetImage.text['des'])

            w, h = AB.size
            w2 = int(w / 2)
            A = AB.crop((0, 0, w2, h))
            B = AB.crop((w2, 0, w, h))
            A = A.resize((self.opt.load_size, self.opt.load_size))
            B = B.resize((self.opt.load_size, self.opt.load_size))
            flash = skimage.img_as_float(B)
            ambient = skimage.img_as_float(A)
            flash = makeBright(flash, 1.2)

            # flash_avg = getAverage(flash[:,:,0]) + getAverage(flash[:,:,1]) + getAverage(flash[:,:,2])
            # ambient_avg = getAverage(ambient[:,:,0]) + getAverage(ambient[:,:,1]) + getAverage(ambient[:,:,2])
            # flash = flash * 2 * ambient_avg / (flash_avg + 0.001)

            # ambient = makeBright(ambient,0.5)
            if des == 21:
                flash = changeTemp(flash, 48, des)
                ambient = changeTemp(ambient, 48, des)

            A = flash + ambient
            B = ambient
            A = xyztorgb(A, des)
            B = xyztorgb(B, des)

        transform_params = get_params(self.opt, A.size)
        # apply the same transform to both A and B
        A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))
        depth_transform = get_transform(self.opt, transform_params, grayscale=True)

        depth_A = Image.fromarray((self.estimateDepth(np.asarray(A)/255)*255).astype('uint8'))
        depth_B = Image.fromarray((self.estimateDepth(np.asarray(B)/255)*255).astype('uint8'))
        # showImage(depth_A)
        # showImage(depth_B)
        depth_A = depth_transform(depth_A)
        depth_B = depth_transform(depth_B)


        A = A_transform(A)
        B = B_transform(B)


        # print(torch.shape(A))
        return {'A': A, 'B': B,'depth_A':depth_A,'depth_B':depth_B, 'A_paths': AB_path, 'B_paths': AB_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)


def xyztorgb(image, des):
    illum = chromaticityAdaptation(des)

    mat = [[3.2404542, -0.9692660, 0.0556434], [-1.5371385, 1.8760108, -0.2040259], [-0.4985314, 0.0415560, 1.0572252]]
    image = np.matmul(image, illum)
    image = np.matmul(image, mat)
    image = np.where(image < 0, 0, image)
    image = np.where(image > 1, 1, image)
    out = (image * 255).astype('uint8')
    out = Image.fromarray(out)
    return out


def xyztolab(image, des):
    illum = chromaticityAdaptation(des)
    out = np.matmul(image, illum)

    out1 = xyz2lab(out).astype(np.float32)
    # print("lab")

    # out2 = (out1 * 255 / np.max(out1)).astype('uint8')

    # out = Image.fromarray(out1)
    return out1


def makeBright(image, ratio):
    image[:, :, 0] = image[:, :, 0] * ratio
    image[:, :, 1] = image[:, :, 1] * ratio
    image[:, :, 2] = image[:, :, 2] * ratio
    return image


def chromaticityAdaptation(calibrationIlluminant):
    if (calibrationIlluminant == 17):
        illum = [[0.8652435, 0.0000000, 0.0000000],
                 [0.0000000, 1.0000000, 0.0000000],
                 [0.0000000, 0.0000000, 3.0598005]]
    elif (calibrationIlluminant == 19):
        illum = [[0.9691356, 0.0000000, 0.0000000],
                 [0.0000000, 1.0000000, 0.0000000],
                 [0.0000000, 0.0000000, 0.9209267]]
    elif (calibrationIlluminant == 20):
        illum = [[0.9933634, 0.0000000, 0.0000000],
                 [0.0000000, 1.0000000, 0.0000000],
                 [0.0000000, 0.0000000, 1.1815972]]
    elif (calibrationIlluminant == 21):
        illum = [[1, 0, 0],
                 [0, 1, 0],
                 [0, 0, 1]]
    elif (calibrationIlluminant == 23):
        illum = [[1.0077340, 0.0000000, 0.0000000],
                 [0.0000000, 1.0000000, 0.0000000],
                 [0.0000000, 0.0000000, 0.8955170]]
    return illum


def getRatio(t, low, high):
    dist = t - low
    range = (high - low) / 100
    return dist / range


def changeTemp(image, tempChange, des):
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
        if t <= 10000 and t > 6500:
            # print("here3")
            r = getRatio(t, 6500, 10000)
            xD = 0.3
            yD = 0.3
            xS = 0.3118
            yS = 0.3224
            r_x = (xD - xS) / 100
            xD = xS + r * r_x
            r_y = (yD - yS) / 100
            yD = yS + r * r_y

        elif 6500 >= t and t > 5500:
            # print("here2")
            r = getRatio(t, 5500, 6500)
            xD = 0.3118
            yD = 0.3224
            xS = 0.3580
            yS = 0.3239
            r_x = (xD - xS) / 100
            xD = xS + r * r_x
            r_y = (yD - yS) / 100
            yD = yS + r * r_y

        elif 5000 <= t and t < 5500:
            # print("here")
            r = getRatio(t, 5500, 5000)
            xD = 0.3752
            yD = 0.3238
            xS = 0.3580
            yS = 0.3239
            r_x = (xD - xS) / 100
            xD = xS + r * r_x
            r_y = (yD - yS) / 100
            yD = yS + r * r_y

        elif (t >= 4500 and t < 5000):
            # print("here6")
            r = getRatio(t, 5000, 4500)

            xD = 0.4231
            yD = 0.3304
            xS = 0.3752
            yS = 0.3238
            r_x = (xD - xS) / 100
            xD = xS + r * r_x
            r_y = (yD - yS) / 100
            yD = yS + r * r_y

        elif t < 4500 and t >= 4000:
            # print("here7")
            r = getRatio(t, 4500, 4000)
            xD = 0.4949
            yD = 0.3564
            xS = 0.4231
            yS = 0.3304
            r_x = (xD - xS) / 100
            xD = xS + r * r_x
            r_y = (yD - yS) / 100
            yD = yS + r * r_y

        elif t >= 0 and t < 4000:
            # print("here9")
            r = getRatio(t, 4000, 0)
            xD = 0.5041
            yD = 0.3334
            xS = 0.4949
            yS = 0.3564
            r_x = (xD - xS) / 100
            xD = xS + r * r_x
            r_y = (yD - yS) / 100
            yD = yS + r * r_y

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
            r = getRatio(t, 4500, 7500)
            xD = 0.17
            yD = 0.17
            xS = 0.4231
            yS = 0.3304
            r_x = (xD - xS) / 100
            xD = xS + r * r_x
            r_y = (yD - yS) / 100
            yD = yS + r * r_y

        elif (t < 4500 and t >= 4000):

            r = getRatio(t, 4500, 4000)
            xD = 0.4949
            yD = 0.3564
            xS = 0.4231
            yS = 0.3304
            r_x = (xD - xS) / 100
            xD = xS + r * r_x
            r_y = (yD - yS) / 100
            yD = yS + r * r_y

        elif (t >= 3500 and t < 4000):

            r = getRatio(t, 4000, 3500)
            xD = 0.5141
            yD = 0.3434
            xS = 0.4949
            yS = 0.3564
            r_x = (xD - xS) / 100
            xD = xS + r * r_x
            r_y = (yD - yS) / 100
            yD = yS + r * r_y
        elif (t > 0 and t < 3500):
            r = getRatio(t, 3500, 0)
            xD = 0.5189
            yD = 0.3063
            xS = 0.5141
            yS = 0.3434
            r_x = (xD - xS) / 100
            xD = xS + r * r_x
            r_y = (yD - yS) / 100
            yD = yS + r * r_y

        chromaticity_x = 0.4231

        chromaticity_y = 0.3304

    offset_x = xD / chromaticity_x
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
    out[:, :, 0] = out0
    out[:, :, 2] = out2
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
def getAverage(ambient_red_bg):
    # ambient_red_bg = ambient_lin[:, :, 0]
    ambient_red_bg = np.reshape(ambient_red_bg, 256 * 256)
    dropOut = int(np.floor(len(ambient_red_bg) / 3))
    # print(len(ambient_red_bg))
    # print(dropOut)
    ambient_red_bg = np.sort(ambient_red_bg)
    ambient_red_bg = np.delete(ambient_red_bg, range(0, dropOut))
    # print(len(ambient_red_bg) - int(np.floor(len(ambient_red_bg)/5)))
    ambient_red_bg = np.delete(ambient_red_bg,
                               range((len(ambient_red_bg) - dropOut), len(ambient_red_bg)))
    avg_ambient_red_bg = np.average(ambient_red_bg)
    return avg_ambient_red_bg


def lin(srgb):
    srgb = srgb.astype(np.float)
    rgb = np.zeros_like(srgb).astype(np.float)
    srgb = srgb
    # print(srgb)
    mask1 = srgb <= 0.04045
    mask2 = (1 - mask1).astype(bool)
    rgb[mask1] = srgb[mask1] / 12.92
    rgb[mask2] = ((srgb[mask2] + 0.055) / 1.055) ** 2.4
    # rgb[mask2] = ((srgb[mask2] + 0.055) / 1.055) ** 2.4
    # rgb[rgb<0.0] = 0.0
    # rgb[rgb>1.0] = 1.0
    # print(rgb)
    rgb = rgb
    # print(rgb)
    return rgb




def gama_corect(rgb):
    srgb = np.zeros_like(rgb)
    mask1 = (rgb > 0) * (rgb < 0.0031308)
    mask2 = (1 - mask1).astype(bool)
    srgb[mask1] = 12.92 * rgb[mask1]
    srgb[mask2] = 1.055 * np.power(rgb[mask2], 0.41666) - 0.055
    srgb[srgb < 0] = 0
    srgb *= 255
    return srgb



def alpha_blend(person, room):
    alpha = person.split()[-1]
    alpha = np.array(alpha)
    alpha = np.reshape(alpha, [256, 256, 1])
    alpha = np.tile(alpha, [1, 1, 3])
    alpha = alpha.astype(float) / 255
    person = np.array(person) / 255
    foreground = person[:, :, 0:3]
    room = np.array(room) / 255
    foreground = foreground.astype(float)
    background = room.astype(float)
    foreground = cv2.multiply(alpha, foreground)
    background = cv2.multiply(1.0 - alpha, background)
    outImage = cv2.add(foreground, background)
    return outImage
