import os
import re
import random
import itertools
import DeepRFT.forward_model as forward_model
import numpy as np
import scipy.ndimage
import glob
import argparse
import PIL
import matplotlib.pyplot as plt
import ReepRFT_utils


def min_max_norm(x):
    x = (x - x.min()) / (x.max() - x.min())
    return x

parser = argparse.ArgumentParser(description='Image Deblurring')
parser.add_argument('--input_dir', default='H:\\GOPRO_Large\\test\\GOPR0410_11_00\\sharp', type=str, help='Directory of validation images')
args = parser.parse_args()


# list of TEST kernels

ker_names = ['ker_07_s=19_i=0.75.png','ker_09_s=19_i=0.25.png',
                'ker_00_s=38_i=0.75.png','ker_07_s=57_i=0.50.png']
noise_list = [0.03]


# traverse thru all kernels & noise levels
# sharp_img_files = glob.glob(os.path.join(args.input_dir, '*.png'))
sharp_img_files = glob.glob(os.path.join(args.input_dir, '*.png'))
for (ker, n) in itertools.product(ker_names, noise_list):
    # load kernel and set noise level
    args.noise = n
    kernel_file = r'H:\GOPRO_Large\blur_kernel' + '\\' + ker
    kernel = np.array(PIL.Image.open(kernel_file)).astype('float32')
    args.kernel = kernel
    ker_subscript = re.findall(r'ker(.*)\.png', kernel_file.split('\\')[-1])[0]
    args.output_dir = os.path.join('\\'.join(args.input_dir.split('\\')[:-1]), 'blur' + ker_subscript + '_n='+str(args.noise))
    os.makedirs(args.output_dir, exist_ok=True)
    model_forward = forward_model.BlurModel(args.kernel, args.noise, 'gaussian', False)  # replace 'gaussian' by 's&p'

    sharp_img_select = random.choices(sharp_img_files, k=25)
    for f in sharp_img_select:
        img = np.array(PIL.Image.open(f)).astype('float32')/255
        # img = np.stack([scipy.ndimage.gaussian_filter(img[...,i], sigma=5.0) for i in range(3)], axis=-1)
        blur_img = model_forward(img)
        # clip to 0-1
        blur_img = np.clip(blur_img, 0, 1)
        # plt.imsave(os.path.join(args.output_dir, f.split('\\')[-1]), (255*min_max_norm(blur_img)).astype('uint8'))
        plt.imsave(os.path.join(args.output_dir, f.split('\\')[-1]), (255*blur_img).astype('uint8'))
        print('Generated blur image %s'%os.path.join(args.output_dir, f.split('\\')[-1]))