import os
import argparse
import glob
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import ReepRFT_utils
from DeepRFT.data_RGB import get_test_data
from DeepRFT.DeepRFT_MIMO import DeepRFT as mynet
from skimage import img_as_ubyte
from DeepRFT.get_parameter_number import get_parameter_number
from tqdm import tqdm
from DeepRFT.layers import *
from skimage.metrics import peak_signal_noise_ratio as psnr_loss
import cv2
import PIL
import scipy.io as sio

import DeepRFT.forward_model as forward_model

wk_dir = r'H:\GOPRO_Large\test\GOPR0410_11_00'
inputds = glob.glob(wk_dir+r'\blur_09_s=19_i=0.25_n=0.03')
targetd = wk_dir+r'\sharp'


# testing parameters
parser = argparse.ArgumentParser(description='Image Deblurring')
# parser.add_argument('--input_dir', default=inputd, type=str, help='Directory of validation images')
parser.add_argument('--target_dir', default=targetd, type=str, help='Directory of validation images')
parser.add_argument('--output_dir', default='H:\\DeepRFT\\GoPro', type=str, help='Directory of validation images')
parser.add_argument('--weights', default='./checkpoints/DeepRFT/model_GoPro.pth', type=str, help='Path to weights')
parser.add_argument('--get_psnr', default=False, type=bool, help='PSNR')
parser.add_argument('--gpus', default='0', type=str, help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--save_result', default=True, type=bool, help='save result')
parser.add_argument('--win_size', default=256, type=int, help='window size, [GoPro, HIDE, RealBlur]=256, [DPDD]=512')
parser.add_argument('--num_res', default=8, type=int, help='num of resblocks, [Small, Med, PLus]=[4, 8, 20]')
parser.add_argument('--max_cycles', default=20, type=int, help='num of restoration-forward cycles')
args = parser.parse_args()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus


# traverse all input folders
for inputd in inputds:
    args.input_dir = inputd
    s_input = inputd.split('\\')
    # load kernel
    kernel_name = "ker_"+('_').join(s_input[-1].split('_')[1:-1])+".png"  # [1:-2] for snp data
    kernel_file = r"H:\GOPRO_Large\blur_kernel\\"+kernel_name
    try:
        kernel = np.array(PIL.Image.open(kernel_file)).astype('float32')
        args.kernel = kernel
    except:
        print("kernel not found")
        continue
    
    # test and save outputs
    result_dir = args.output_dir + str(args.input_dir).split('\\')[-2][4:]+str(args.input_dir).split('\\')[-1][4:]
    win = args.win_size
    get_psnr = args.get_psnr
    # model_restoration = mynet()
    model_restoration = mynet(num_res=args.num_res, inference=True)
    # print number of model
    get_parameter_number(model_restoration)
    # utils.load_checkpoint(model_restoration, args.weights)
    ReepRFT_utils.load_checkpoint_compress_doconv(model_restoration, args.weights)
    print("===>Testing using weights: ", args.weights)
    model_restoration.cuda()
    model_restoration = nn.DataParallel(model_restoration)
    model_restoration.eval()

    # forward model
    model_forward = forward_model.BlurModel(args.kernel)

    test_dataset = get_test_data(args.input_dir, args.target_dir, img_options={})
    test_loader  = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, drop_last=False, pin_memory=True)
    psnr_val_rgb = []
    psnr = 0

    ReepRFT_utils.mkdir(result_dir)

    with torch.no_grad():
        psnr_list = []
        ssim_list = []
        for ii, (test_inp, test_tar, filenames) in enumerate(tqdm(test_loader), 0):
            if filenames != ['000194']:
                continue

            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()
            # input_    = data_test[0].cuda()
            # filenames = data_test[1]
            input_ = test_inp
            _, _, Hx, Wx = input_.shape
            # filenames = data_test[1]

            xx_list = [input_.cpu().numpy().squeeze()]
            im_list = []
            for i in range(args.max_cycles):
                # partition inputs into patches
                input_re, batch_list = window_partitionx(input_, win)
                # restore
                restored = model_restoration(input_re)
                # un-partition patches
                restored = window_reversex(restored, win, Hx, Wx, batch_list)
                # pass thru forward model
                input_ = model_forward(restored)
                # record xx and im
                xx_list.append(input_.cpu().numpy().squeeze())
                im_list.append(restored.cpu().numpy().squeeze())

                restored = torch.clamp(restored, 0, 1)
                restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()
                for batch in range(len(restored)):
                    restored_img = restored[batch]
                    restored_img = img_as_ubyte(restored[batch])

                    if args.save_result:
                        ReepRFT_utils.save_img((os.path.join(result_dir, filenames[batch]+'_%d.png'%i)), restored_img)
            test_tar = img_as_ubyte(test_tar.squeeze().permute(1,2,0).cpu().detach().numpy())
            ReepRFT_utils.save_img((os.path.join(result_dir, filenames[batch]+'_target.png')), test_tar)
            sio.savemat(os.path.join(result_dir, filenames[batch]+'.mat'), {'xx_list':xx_list, 'im_list':im_list, 'yy':test_tar})

