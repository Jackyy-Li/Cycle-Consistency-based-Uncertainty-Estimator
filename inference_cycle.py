import argparse
import glob
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt
import scipy.io as sio
from PIL import Image

from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact
from Pooling import Cycle
from ReepRFT_utils import noise, random_crop


def main():
    """Inference demo for Real-ESRGAN.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str,
                        default='I:\\DIV2K\\X4_bicubic\\valid', help='Input image or folder')
    parser.add_argument(
        '-n',
        '--model_name',
        type=str,
        default='RealESRGAN_x4plus',
        help=('Model names: RealESRGAN_x4plus | RealESRNet_x4plus | RealESRGAN_x4plus_anime_6B | RealESRGAN_x2plus | '
              'realesr-animevideov3 | realesr-general-x4v3'))
    parser.add_argument('-o', '--output', type=str,
                        default='results', help='Output folder')
    parser.add_argument(
        '-dn',
        '--denoise_strength',
        type=float,
        default=0,
        help=('Denoise strength. 0 for weak denoise (keep noise), 1 for strong denoise ability. '
              'Only used for the realesr-general-x4v3 model'))
    parser.add_argument('-s', '--outscale', type=float, default=4,
                        help='The final upsampling scale of the image')
    parser.add_argument('-cn', '--cycle_number', type=int, default=20, help='Number of cycles for the inference process')
    parser.add_argument('-nl', '--noise_level', type=float, default=0, help='Add Noise level for the inference process')
    parser.add_argument('-k', '--kernel_size', type=int, default=4, help='Kernel size for the inference process')
    parser.add_argument('--crop_ratio', type=int, default=1, help='crop_ratio for the inference process')
    parser.add_argument(
        '-p', '--model_path', type=str, default=None, help='[Option] Model path. Usually, you do not need to specify it')
    parser.add_argument('--suffix', type=str, default='',
                        help='Suffix of the restored image')
    parser.add_argument('-t', '--tile', type=int, default=0,
                        help='Tile size, 0 for no tile during testing')
    parser.add_argument('--tile_pad', type=int,
                        default=10, help='Tile padding')
    parser.add_argument('--pre_pad', type=int, default=0,
                        help='Pre padding size at each border')
    parser.add_argument('--face_enhance', action='store_true',
                        help='Use GFPGAN to enhance face')
    parser.add_argument(
        '--fp32', action='store_true', help='Use fp32 precision during inference. Default: fp16 (half precision).')
    parser.add_argument(
        '--alpha_upsampler',
        type=str,
        default='realesrgan',
        help='The upsampler for the alpha channels. Options: realesrgan | bicubic')
    parser.add_argument(
        '--ext',
        type=str,
        default='auto',
        help='Image extension. Options: auto | jpg | png, auto means using the same extension as inputs')
    parser.add_argument(
        '-g', '--gpu-id', type=int, default=None, help='gpu device to use (default=None) can be 0,1,2 for multi-gpu')

    args = parser.parse_args()

    # determine models according to model names
    args.model_name = args.model_name.split('.')[0]
    if args.model_name == 'RealESRGAN_x4plus':  # x4 RRDBNet model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                        num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
        file_url = [
            'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth']
    elif args.model_name == 'RealESRNet_x4plus':  # x4 RRDBNet model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                        num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
        file_url = [
            'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth']
    elif args.model_name == 'RealESRGAN_x4plus_anime_6B':  # x4 RRDBNet model with 6 blocks
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                        num_block=6, num_grow_ch=32, scale=4)
        netscale = 4
        file_url = [
            'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth']
    elif args.model_name == 'RealESRGAN_x2plus':  # x2 RRDBNet model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                        num_block=23, num_grow_ch=32, scale=2)
        netscale = 2
        file_url = [
            'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth']
    # x4 VGG-style model (XS size)
    elif args.model_name == 'realesr-animevideov3':
        model = SRVGGNetCompact(
            num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')
        netscale = 4
        file_url = [
            'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth']
    # x4 VGG-style model (S size)
    elif args.model_name == 'realesr-general-x4v3':
        model = SRVGGNetCompact(
            num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
        netscale = 4
        file_url = [
            'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth',
            'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth'
        ]

    # determine model paths
    if args.model_path is not None:
        model_path = args.model_path
    else:
        model_path = os.path.join('weights', args.model_name + '.pth')
        if not os.path.isfile(model_path):
            ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
            for url in file_url:
                # model_path will be updated
                model_path = load_file_from_url(
                    url=url, model_dir=os.path.join(ROOT_DIR, 'weights'), progress=True, file_name=None)

    # use dni to control the denoise strength
    dni_weight = None
    if args.model_name == 'realesr-general-x4v3' and args.denoise_strength != 1:
        wdn_model_path = model_path.replace(
            'realesr-general-x4v3', 'realesr-general-wdn-x4v3')
        model_path = [model_path, wdn_model_path]
        dni_weight = [args.denoise_strength, 1 - args.denoise_strength]

    # restorer
    upsampler = RealESRGANer(
        scale=netscale,
        model_path=model_path,
        dni_weight=dni_weight,
        model=model,
        tile=args.tile,
        tile_pad=args.tile_pad,
        pre_pad=args.pre_pad,
        half=not args.fp32,
        gpu_id=args.gpu_id)

    if args.face_enhance:  # Use GFPGAN for face enhancement
        from gfpgan import GFPGANer
        face_enhancer = GFPGANer(
            model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
            upscale=args.outscale,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=upsampler)
    os.makedirs(args.output, exist_ok=True)

    if os.path.isfile(args.input):
        paths = [args.input]
    else:

        # image changed but path stays the same
        paths = sorted(glob.glob(os.path.join(args.input, '*')))


    '''
    model path:
    face_model: .\experiments\finetune_RealESRGANx4plus_400k_face\models\net_g_latest.pth
    microscopy_model: .\experiments\finetune_RealESRGANx4plus_400k_micro\models\net_g_latest.pth
    '''


    # parameters of cycle inference
    cycle_number = args.cycle_number    # number of cycles
    noise_level = args.noise_level      # noise level
    kernel_size = args.kernel_size      # kernel size
    crop_ratio = args.crop_ratio          # crop size


    # setup of .mat file
    target_dir = args.output
    xx_list = []  # x, x1 - input images for each cycle
    im_list = []  # y0, y1，- output images for each cycle
    yy = np.empty((3,3,3))  # ground truth: later assgined to intial image
    x = np.empty((3,3,3))  # initial image: later assgined to intial image


    # first downscale cycle to generate x0
    for idx, path in enumerate(paths):
        imgname, extension = os.path.splitext(os.path.basename(path))

        # random_crop the image
        if crop_ratio > 1:
            img = random_crop(plt.imread(args.input + "\\" + imgname + extension), crop_ratio)

        img = plt.imread(args.input + "\\" + imgname + extension)

        # store the ground truth (yy)
        yy = img

        # downscale the image to x0
        print('generating x0')
        lr_cycle = Cycle(target_dir, imgname, kernel_size)
        img = lr_cycle.apply_avgpooling_cycle(yy)  # x, numpy ndarray

        # introduce random noise toward x
        if noise_level > 0:
            img = noise(img, noise_level)

        # store x to xx_list as initial input
        xx_list.append(img)

        for i in range(cycle_number):
            # Upsample the image to y
            print('upsampling', idx, imgname)
            img = np.uint8(img * 255) # convert to uint8

            if len(img.shape) == 3 and img.shape[2] == 4:
                img_mode = 'RGBA'
            else:
                img_mode = None
            try:
                if args.face_enhance:
                    _, _, output = face_enhancer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
                else:
                    output, _ = upsampler.enhance(img, outscale=args.outscale)
            except RuntimeError as error:
                print('Error', error)
                print('If you encounter CUDA out of memory, try to set --tile with a smaller number.')
            else:
                if args.ext == 'auto':
                    extension = extension[1:]
                else:
                    extension = args.ext
                if img_mode == 'RGBA':  # RGBA images should be saved in png format
                    extension = 'png'
                if args.suffix == '':
                    # the output file name will be the same as the input file name
                    save_path = os.path.join(args.output, f'{imgname}.{extension}')
                    cycle_result = os.path.join(target_dir, f'{imgname}_{i}.{extension}')
                else:
                    # made change to let save—path be the same as the original path to accumlate the results
                    save_path = os.path.join(args.output, f'{imgname}_{args.suffix}.{extension}')

                # plt.imsave(cycle_result, output)    # save the cycle_result image
                im_list.append(np.array(output))    # save the upscaled image to yy_list

            # downscale the image to x
            print("downsampling", idx, imgname)
            extension = ".png"  # keep extension as png to be consistent with the original image
            img = lr_cycle.apply_avgpooling_cycle(output)
            x_dir = os.path.join(target_dir, f'{imgname}_x{i}.{extension}')
            # plt.imsave(x_dir, img[:,:,:3])
            xx_list.append(img)

        # save yy, xx_list and im_list to mat file for each picture
        sio.savemat(os.path.join(target_dir, f'cycle_{imgname}.mat'),  {'xx_list': xx_list, 'im_list': im_list, 'yy': yy})

        # set xx_list and im_list to empty for the next picture
        xx_list = []
        im_list = []


if __name__ == '__main__':
    main()
