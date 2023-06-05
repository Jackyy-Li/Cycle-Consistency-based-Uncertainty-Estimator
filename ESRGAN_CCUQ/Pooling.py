import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


# pooling class performs pooling on an input HR image and outputs an LR image
# It has two method function: maxpooling2d() and avgpooling2d()

# 1. maxpooling2d() performs maxpooling on the input HR image
# outputs an LR image using the maxpooling2d() function in torch.nn.functional

# 2. avgpooling2d() performs avgpooling on the input HR image
# outputs an LR image using the avgpooling2d() function in torch.nn.functional

# Kernel Size: the size of the window to take a max over (2 x 2)
# In pooling class, kernel size == stride size == upsampling size of ESRGAN

'''input of size = [N,C,H, W] or [C,H, W] for maxpooling2d() and avgpooling2d()
N==>batch size,
C==> number of channels,
H==> height of input planes in pixels,
W==> width in pixels.
'''


class Cycle():
    def __init__(self, target_dir, imgname, kernel_size=4):
        self.target_dir = target_dir
        self.kernel_size = kernel_size
        # self.image_tensor = image_tensor
        self.imgname = imgname

    def apply_maxpooling_cycle(self, image):
        # Convert the output LR image back to PIL image
        image_tensor = T.ToTensor()(image)
        max_pool_image = maxpooling2d(image_tensor, self.kernel_size).numpy().transpose((1, 2, 0))
        plt.imsave(self.target_dir + "\\" + self.imgname + '.png', max_pool_image[:,:,:3])
        return max_pool_image

    def apply_avgpooling_cycle(self, image):
        # apply avgpooling on the image tensors
        # return: [H, W, C]
        image_tensor = T.ToTensor()(image)
        avg_pool_image = avgpooling2d(image_tensor, self.kernel_size).numpy().transpose((1, 2, 0))
        plt.imsave(self.target_dir + "\\" + self.imgname + '.png', avg_pool_image[:,:,:3])
        return avg_pool_image


def maxpooling2d(img, kernel_size=4):
    stride_size = kernel_size
    input_img = img.unsqueeze(0)

    # implement max_pooling
    max_pool_img = F.max_pool2d(input_img, kernel_size, stride_size)
    max_pool_img = max_pool_img.squeeze(0)  # restore the image to the original size
    return max_pool_img

def avgpooling2d(img, kernel_size=4):
    stride_size = kernel_size
    input_img = img.unsqueeze(0)

    # implement avg_pooling
    avg_pool_img = F.avg_pool2d(input_img, kernel_size, stride_size)
    avg_pool_img = avg_pool_img.squeeze(0)  # restore the image to the original size
    return avg_pool_img

