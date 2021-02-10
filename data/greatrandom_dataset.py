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
import time


import matplotlib.pyplot as plt
def showImage(img,title=None):
    plt.imshow(img, cmap= 'inferno')
    plt.colorbar()
    if title is not None:
        plt.title(title)
    plt.show()

class GreatRandomDataset(BaseDataset):

    def __init__(self, opt):

        BaseDataset.__init__(self, opt)

        # 10000 is the max dataset size
        if opt.phase == 'test':
            self.dir_ourdataset = os.path.join(opt.dataroot, 'our_dataset2_test')
            self.images_dir_ourdataset = sorted(make_dataset(self.dir_ourdataset + '/amb_0.5', 100000))

            self.dir_multidataset = os.path.join(opt.dataroot, 'multi_dataset_test')
            self.images_dir_multidataset = sorted(make_dataset(self.dir_multidataset + '/amb_0.5/1', 100000))
            # self.images_dir_multidataset = self.images_dir_multidataset * 4

            self.dir_portraitdataset = os.path.join(opt.dataroot, 'portrait_dataset_extra_test')
            self.images_dir_portraitdataset = sorted(make_dataset(self.dir_portraitdataset+'/amb_0.5/1', 100000))
            # self.images_dir_portraitdataset =  self.images_dir_portraitdataset*4
        else:
            self.dir_ourdataset = os.path.join(opt.dataroot,'our_dataset2')
            self.images_dir_ourdataset = sorted(make_dataset(self.dir_ourdataset+'/amb_0.5', 100000 ))

            self.dir_multidataset = os.path.join(opt.dataroot,'multi_dataset_complete_png')
            self.images_dir_multidataset = sorted(make_dataset(self.dir_multidataset+'/amb_0.5/1', 100000 ))
            self.images_dir_multidataset =  self.images_dir_multidataset*2

            self.dir_portraitdataset = os.path.join(opt.dataroot, 'portrait_dataset_png')
            self.images_dir_portraitdataset = sorted(make_dataset(self.dir_portraitdataset+'/amb_0.5/1', 100000))
            self.images_dir_portraitdataset =  self.images_dir_portraitdataset*4


        self.images_dir_all = self.images_dir_ourdataset + self.images_dir_multidataset + self.images_dir_portraitdataset

        self.data_size = opt.load_size
        self.data_root = opt.dataroot

        # opt_merge = copy.deepcopy(opt)
        # opt_merge.isTrain = False
        # opt_merge.model = 'pix2pix4depth'
        # self.mergenet = Pix2Pix4DepthModel(opt_merge)
        # self.mergenet.save_dir = 'depthmerge/checkpoints/scaled_04_1024'
        # self.mergenet.load_networks('latest')
        # self.mergenet.eval()
        #
        # self.device = torch.device('cuda:0')
        #
        # midas_model_path = "midas/model-f46da743.pt"
        # self.midasmodel = MidasNet(midas_model_path, non_negative=True)
        # self.midasmodel.to(self.device)
        # self.midasmodel.eval()

        # torch.multiprocessing.set_start_method('spawn')
        #
        # for i in range(len(self.images_dir_all)):
        #     self.__getitem__(i)

    def __getitem__(self, index):
        image_path_temp = self.images_dir_all[index]
        image_name = image_path_temp.split('/')[-1]

        amb_select = random.randint(0, 2)
        if amb_select == 0:
            amb_dir = '/amb_0.5'
        elif amb_select == 1:
            amb_dir = '/amb_0.75'
        elif amb_select == 2:
            amb_dir = '/amb_1'
        else:
            print(amb_select)
            exit()


        if self.opt.phase == 'test':
            if 'our_dataset2_test/' in image_path_temp:
                image_path = self.data_root + '/our_dataset2_test' + amb_dir + '/{}'.format(image_name)
            elif 'multi_dataset_test/' in image_path_temp:
                multi_select = random.randint(1, 10)
                image_path = self.data_root + '/multi_dataset_test' + amb_dir + '/{}'.format(multi_select) + '/{}'.format(
                    image_name)
            elif 'portrait_dataset_extra_test/' in image_path_temp:
                portrait_select = random.randint(1, 20)
                image_path = self.data_root + '/portrait_dataset_extra_test' + amb_dir + '/{}'.format(portrait_select) + '/{}'.format(
                image_name)
        else:
            if 'our_dataset2/' in image_path_temp:
                image_path = self.data_root + '/our_dataset2' + amb_dir + '/{}'.format(image_name)
            elif 'multi_dataset_complete_png/' in image_path_temp:
                multi_select = random.randint(1, 10)
                image_path = self.data_root + '/multi_dataset_complete_png' + amb_dir + '/{}'.format(multi_select) + '/{}'.format(image_name)
            elif 'portrait_dataset_png/' in image_path_temp:
                portrait_select = random.randint(1, 20)
                image_path = self.data_root + '/portrait_dataset_png' + amb_dir + '/{}'.format(portrait_select) + '/{}'.format(
                    image_name)

        if 'our_dataset' in image_path:
            image_pair = Image.open(image_path)
            hyper_des = int(PngImageFile(image_path).text['des'])
            temp_select = random.randint(0, 2)

            A,B = self.divide_imagepair(image_pair)
            ambient = skimage.img_as_float(A)
            flash = skimage.img_as_float(B)
            # print("here")
            # if hyper_des == 17:
            #     if temp_select == 0:
            #         ambient = self.changeTemp(ambient, 44, hyper_des)
            #     elif temp_select == 1:
            #         ambient = self.changeTemp(ambient, 54, hyper_des)
            if hyper_des == 21:
                flash = self.changeTemp(flash, 48, hyper_des)
                ambient = self.changeTemp(ambient, 48, hyper_des)
                # if temp_select == 0:
                #     ambient = self.changeTemp(ambient, 44, hyper_des)
                # elif temp_select == 1:
                #     ambient = self.changeTemp(ambient, 54, hyper_des)
                # elif temp_select == 2:
            flashPhoto = flash + ambient
            flashPhoto[flashPhoto < 0] = 0
            flashPhoto[flashPhoto > 1] = 1
            flashPhoto = self.xyztorgb(flashPhoto, hyper_des)
            ambient = self.xyztorgb(ambient, hyper_des)

            flashphoto_depth = self.getDepth(flashPhoto,image_path,'flash')
            ambient_depth = self.getDepth(ambient,image_path,'ambient')


        elif 'multi_dataset' in image_path:
            image_pair = Image.open(image_path)

            A, B = self.divide_imagepair(image_pair)
            ambient = skimage.img_as_float(A)
            flash = skimage.img_as_float(B)
            temp_select = random.randint(0, 2)
            # if temp_select == 0:
            #     ambient = self.changeTemp(ambient, 1, 17)
            # elif temp_select == 1:
            #     ambient = self.changeTemp(ambient, 2, 17)
            flashPhoto = flash + ambient
            flashPhoto[flashPhoto < 0] = 0
            flashPhoto[flashPhoto > 1] = 1
            ambient = Image.fromarray((ambient*255).astype('uint8'))
            flashPhoto = Image.fromarray((flashPhoto*255).astype('uint8'))

            flashphoto_depth = self.getDepth(flashPhoto, image_path, 'flash')
            ambient_depth = self.getDepth(ambient, image_path, 'ambient')

        elif 'portrait_dataset' in image_path:
            image_pair = Image.open(image_path)

            A, B = self.divide_imagepair(image_pair)
            ambient = skimage.img_as_float(A)
            flash = skimage.img_as_float(B)

            ambient = self.lin(ambient)
            flash = self.lin(flash)

            flashPhoto = flash + ambient
            flashPhoto[flashPhoto < 0] = 0
            flashPhoto[flashPhoto > 1] = 1
            ambient = Image.fromarray((ambient * 255).astype('uint8'))
            flashPhoto = Image.fromarray((flashPhoto * 255).astype('uint8'))

            flashphoto_depth = self.getDepth(flashPhoto, image_path, 'flash')
            ambient_depth = self.getDepth(ambient, image_path, 'ambient')

        torch.cuda.empty_cache()

        ambient_orgsize = skimage.img_as_float(ambient)
        flashPhoto_orgsize = skimage.img_as_float(flashPhoto)

        ambient = ambient.resize((self.data_size, self.data_size))
        flashPhoto = flashPhoto.resize((self.data_size, self.data_size))
        ambient_depth = ambient_depth.resize((self.data_size, self.data_size))
        flashphoto_depth = flashphoto_depth.resize((self.data_size, self.data_size))

        transform_params = get_params(self.opt, ambient.size)
        rgb_transform = get_transform(self.opt, transform_params, grayscale=False)
        depth_transform = get_transform(self.opt, transform_params, grayscale=True)

        ambient = rgb_transform(ambient)
        flashPhoto = rgb_transform(flashPhoto)

        ambient_depth = depth_transform(ambient_depth)
        flashphoto_depth = depth_transform(flashphoto_depth)

        return {'A': flashPhoto, 'B': ambient,'A_org':flashPhoto_orgsize,'B_org':ambient_orgsize, 'depth_A': flashphoto_depth, 'depth_B': ambient_depth, 'A_paths': image_path, 'B_paths': image_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.images_dir_all)

    def getDepth(self,image,image_path,id):
        image_path_beheaded = image_path.replace('.png','')
        image_path_beheaded = image_path_beheaded.replace('.jpg','')
        image_path_beheaded = image_path_beheaded.replace('.jpeg','')

        depth_path = image_path_beheaded[:len(self.data_root)]+'/depth'+image_path_beheaded[len(self.data_root):] + '_' + id +'.png'
        if os.path.exists(depth_path):
            depth = Image.open(depth_path)
        else:
            depth = self.estimateDepth(np.asarray(image)/255)
            depth_dir = depth_path.replace(depth_path.split('/')[-1],'')
            if not os.path.exists(depth_dir):
                os.makedirs(depth_dir)
            depth = (depth*255).astype('uint8')
            cv2.imwrite(depth_path, depth)
            depth = Image.fromarray(depth)
            # depth.save(depth_path)
            print('Depth file cached |',depth_path)

        return depth

    def divide_imagepair(self,image_pair):
        w, h = image_pair.size
        w2 = int(w / 2)
        A = image_pair.crop((0, 0, w2, h))
        B = image_pair.crop((w2, 0, w, h))
        # A = A.resize((self.data_size, self.data_size))
        # B = B.resize((self.data_size, self.data_size))
        return A,B

    def gama_corect(self, rgb):
        srgb = np.zeros_like(rgb)
        mask1 = (rgb > 0) * (rgb < 0.0031308)
        mask2 = (1 - mask1).astype(bool)
        srgb[mask1] = 12.92 * rgb[mask1]
        srgb[mask2] = 1.055 * np.power(rgb[mask2], 0.41666) - 0.055
        srgb[srgb < 0] = 0
        return srgb

    def doubleestimate(self, img, size1, size2):
        estimate1 = self.singleestimate(img, size1)
        estimate1 = cv2.resize(estimate1, (1024, 1024), interpolation=cv2.INTER_CUBIC)

        estimate2 = self.singleestimate(img, size2)
        estimate2 = cv2.resize(estimate2, (1024, 1024), interpolation=cv2.INTER_CUBIC)

        self.mergenet.set_input(estimate1, estimate2)
        self.mergenet.test()
        torch.cuda.empty_cache()
        visuals = self.mergenet.get_current_visuals()
        prediction_mapped = visuals['fake_B']
        prediction_mapped = (prediction_mapped + 1) / 2
        prediction_mapped = (prediction_mapped - torch.min(prediction_mapped)) / (
                torch.max(prediction_mapped) - torch.min(prediction_mapped))
        prediction_mapped = prediction_mapped.squeeze().cpu().numpy()

        prediction_end_res = cv2.resize(prediction_mapped, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)

        return prediction_end_res

    def singleestimate(self, img, msize):
        return self.estimateMidas(img, msize)

    def estimateMidas(self, img, msize):
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
            torch.cuda.empty_cache()

        prediction = prediction.squeeze().cpu().numpy()
        prediction = cv2.resize(prediction, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)

        depth_min = prediction.min()
        depth_max = prediction.max()

        if depth_max - depth_min > np.finfo("float").eps:
            prediction = (prediction - depth_min) / (depth_max - depth_min)
        else:
            prediction = 0

        return prediction

    def estimateDepth(self, rgb_mix):
        rgb_mix = self.gama_corect(rgb_mix)
        depth_temp = self.doubleestimate(rgb_mix, 384, 768)
        return depth_temp

    def xyztorgb(self, image, des):
        illum = self.chromaticityAdaptation(des)
        mat = [[3.2404542, -0.9692660, 0.0556434], [-1.5371385, 1.8760108, -0.2040259],
               [-0.4985314, 0.0415560, 1.0572252]]
        image = np.matmul(image, illum)
        image = np.matmul(image, mat)
        image = np.where(image < 0, 0, image)
        image = np.where(image > 1, 1, image)
        out = (image * 255).astype('uint8')
        out = Image.fromarray(out)
        return out

    def chromaticityAdaptation(self, calibrationIlluminant):
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

    def getRatio(self, t, low, high):
        dist = t - low
        range = (high - low) / 100
        return dist / range

    def changeTemp(self, image, tempChange, des):
        if (des == 17):
            t1 = 5500
            if tempChange == 1:
                tempChange = -800
            elif tempChange == 2:
                tempChange = 1500
            elif tempChange == 44:
                tempChange = -400
            elif tempChange == 40:
                tempChange = - 450
            elif tempChange == 52:
                tempChange = 234
            elif tempChange == 54:
                tempChange = 400
            t = t1 + tempChange
            if t <= 10000 and t > 6500:
                r = self.getRatio(t, 6500, 10000)
                xD = 0.3
                yD = 0.3
                xS = 0.3118
                yS = 0.3224
                r_x = (xD - xS) / 100
                xD = xS + r * r_x
                r_y = (yD - yS) / 100
                yD = yS + r * r_y

            elif 6500 >= t and t > 5500:
                r = self.getRatio(t, 5500, 6500)
                xD = 0.3118
                yD = 0.3224
                xS = 0.3580
                yS = 0.3239
                r_x = (xD - xS) / 100
                xD = xS + r * r_x
                r_y = (yD - yS) / 100
                yD = yS + r * r_y

            elif 5000 <= t and t < 5500:
                r = self.getRatio(t, 5500, 5000)
                xD = 0.3752
                yD = 0.3238
                xS = 0.3580
                yS = 0.3239
                r_x = (xD - xS) / 100
                xD = xS + r * r_x
                r_y = (yD - yS) / 100
                yD = yS + r * r_y

            elif (t >= 4500 and t < 5000):
                r = self.getRatio(t, 5000, 4500)
                xD = 0.4231
                yD = 0.3304
                xS = 0.3752
                yS = 0.3238
                r_x = (xD - xS) / 100
                xD = xS + r * r_x
                r_y = (yD - yS) / 100
                yD = yS + r * r_y

            elif t < 4500 and t >= 4000:
                r = self.getRatio(t, 4500, 4000)
                xD = 0.4949
                yD = 0.3564
                xS = 0.4231
                yS = 0.3304
                r_x = (xD - xS) / 100
                xD = xS + r * r_x
                r_y = (yD - yS) / 100
                yD = yS + r * r_y

            elif t >= 0 and t < 4000:
                r = self.getRatio(t, 4000, 0)
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
                r = self.getRatio(t, 4500, 7500)
                xD = 0.17
                yD = 0.17
                xS = 0.4231
                yS = 0.3304
                r_x = (xD - xS) / 100
                xD = xS + r * r_x
                r_y = (yD - yS) / 100
                yD = yS + r * r_y

            elif (t < 4500 and t >= 4000):
                r = self.getRatio(t, 4500, 4000)
                xD = 0.4949
                yD = 0.3564
                xS = 0.4231
                yS = 0.3304
                r_x = (xD - xS) / 100
                xD = xS + r * r_x
                r_y = (yD - yS) / 100
                yD = yS + r * r_y

            elif (t >= 3500 and t < 4000):
                r = self.getRatio(t, 4000, 3500)
                xD = 0.5141
                yD = 0.3434
                xS = 0.4949
                yS = 0.3564
                r_x = (xD - xS) / 100
                xD = xS + r * r_x
                r_y = (yD - yS) / 100
                yD = yS + r * r_y
            elif (t > 0 and t < 3500):
                r = self.getRatio(t, 3500, 0)
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

        return out

    def lin(self, srgb):
        srgb = srgb.astype(np.float)
        rgb = np.zeros_like(srgb).astype(np.float)
        srgb = srgb
        mask1 = srgb <= 0.04045
        mask2 = (1 - mask1).astype(bool)
        rgb[mask1] = srgb[mask1] / 12.92
        rgb[mask2] = ((srgb[mask2] + 0.055) / 1.055) ** 2.4
        rgb = rgb
        return rgb
