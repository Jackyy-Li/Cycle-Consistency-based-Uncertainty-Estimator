import torch
import torch.nn.functional as F
import numpy as np
import skimage.util

class BlurModel():
    def __init__(self, kernel, noise=None, noise_type='gaussian', cuda=True):
        self.kernel = torch.from_numpy(kernel)
        self.kernel /= self.kernel.mean()
        if self.kernel.ndim == 2:
            self.kernel = self.kernel.unsqueeze(0)
        elif self.kernel.ndim == 3:
            self.kernel = torch.permute(self.kernel, [2,0,1])
        if self.kernel.shape[0] == 1:
            tmp = torch.zeros([3,3,self.kernel.shape[-2], self.kernel.shape[-1]])
            for i in range(3):
                tmp[i,i:i+1,...] = self.kernel
            self.kernel = tmp
        elif self.kernel.shape[0] == 3:
            self.kernel = torch.tile(self.kernel, [3,1,1,1])
        elif self.kernel.shape[0] == 4:  # remove alpha channel
            self.kernel = self.kernel[:3,...]
            self.kernel = torch.tile(self.kernel, [3,1,1,1])
        self.kernel /= torch.sum(self.kernel, dim=(1,2,3), keepdim=True)
        self.cuda = cuda
        if self.cuda:
            self.kernel = self.kernel.cuda()
        self.noise = noise
        if noise_type == 'gaussian' or noise_type == 'Gaussian':
            self.noise_type = 'gaussian'
        elif noise_type == 's&p' or noise_type == 'salt&pepper':
            self.noise_type = 's&p'
        else:
            raise ValueError("Noise must be either Gaussian or S&P.")

    def __call__(self, img):
        if isinstance(img, np.ndarray):
            self.numpy = True
            if img.ndim < 4:
                img = np.expand_dims(img.transpose([2,0,1]), axis=0)  # [N, C, H, W]
            img = torch.from_numpy(img)
        else:
            self.numpy = False
        
        img_c = F.conv2d(img, self.kernel, bias=None, stride=1, padding='same')
        if self.numpy:
            img_c = img_c.numpy().squeeze().transpose([1,2,0])
            
        if self.noise is not None:
            # img_c += np.random.normal(0, self.noise, img_c.shape)
            if self.noise_type == 'gaussian':  # Gaussian additive noise
                img_c = skimage.util.random_noise(img_c, mode=self.noise_type, var=self.noise**2)
            else:  # s&p
                img_c = skimage.util.random_noise(img_c, mode=self.noise_type, amount=self.noise)
            
        return img_c