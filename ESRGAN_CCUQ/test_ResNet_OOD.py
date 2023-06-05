import pandas as pd
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.io as sio
from numpy.linalg import inv
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import average_precision_score
from sklearn.metrics import accuracy_score
import os, re, glob
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import torch.nn as nn
import torch.nn.functional as F


class MixOODDataset(torch.utils.data.Dataset):
    def __init__(self, id_path, ood_path=None, transform=None):
        self.id_path = id_path
        self.ood_path = ood_path
        self.transform = transform

        if id_path is None:
            id_imgs = []
        else:
            id_imgs = glob.glob(os.path.join(id_path, '*_0.png'))
        if ood_path is None:
            ood_imgs = []
        else:
            ood_imgs = glob.glob(os.path.join(ood_path, '*_0.png'))

        self.id_imgs = id_imgs
        self.ood_imgs = ood_imgs

        # Add the random crop and resize transform
        self.random_crop = transforms.RandomCrop((224, 224))
        self.resize = transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR)

    def __len__(self):
        return len(self.id_imgs) + len(self.ood_imgs)

    def __getitem__(self, idx):
        if idx < len(self.id_imgs):
            img_path = self.id_imgs[idx]
            label = 0
        else:
            img_path = self.ood_imgs[idx - len(self.id_imgs)]
            label = 1

        y0 = plt.imread(img_path).transpose([2,0,1])  # [C=3, H, W]
        img = y0
        img = torch.from_numpy(img).float()

        # Convert to 3 channels (RGB) if the image has 4 channels
        if img.shape[0] == 4:
            img = img[:3, :, :]

        # Apply the random crop and resize transform if necessary
        if img.shape[1] >= 224 and img.shape[2] >= 224:
            img = self.random_crop(img)
        else:
            img = self.resize(img)

        if self.transform:
            img = self.transform(img)

        return img, label


DATA_PATH = 'C:\\Jacky\\datasets\\test_data'
DS_LIST = ['anime', 'micro', 'face']
MODEL_LIST = ['anime', 'micro', 'face']


def inference(resnet, dl):
    y_prob_list = []
    y_list = []
    with torch.no_grad():
        resnet.eval()
        valid_loss = 0
        for i, (x, y) in enumerate(dl):
            x = x.cuda()
            y = y.cuda()
            y_pred = resnet(x)
            y_prob_list.append(F.softmax(y_pred, dim=1)[:,1].cpu().numpy())
            y_list.append(y.cpu().numpy())
    y_prob_test = np.concatenate(y_prob_list, axis=0)
    y_pred_test = (y_prob_test > 0.5).astype(int)
    y_test = np.concatenate(y_list, axis=0)

    # Evaluation
    # overall accuracy, ROCAUC, AP
    acc_all_base = accuracy_score(y_test, y_pred_test)
    ap_all_base = average_precision_score(y_test, y_prob_test)
    fpr, tpr, thresholds = roc_curve(y_test, y_prob_test)
    roc_all_base = auc(fpr, tpr)
    return acc_all_base, ap_all_base, roc_all_base


acc_list, ap_list, roc_list = np.zeros([len(MODEL_LIST), len(DS_LIST)]), np.zeros([len(MODEL_LIST), len(DS_LIST)]), np.zeros([len(MODEL_LIST), len(DS_LIST)])

for i in range(len(MODEL_LIST)):
    # Model
    resnet = torch.hub.load('pytorch/vision:v0.14.1', 'resnet50', pretrained=False).cuda()
    resnet.load_state_dict(torch.load('./ResNet_OOD_ckpt/Resnet50_%s_best.pth'%MODEL_LIST[i]))

    for j in range(len(DS_LIST)):
        test_path = os.path.join(DATA_PATH, '%s_dataset'%DS_LIST[j], '%s_cycle_%s'%(MODEL_LIST[i], DS_LIST[j]))
        if i == j:  # ID
            test_ds = MixOODDataset(test_path, None)
        else:  # OOD
            test_ds = MixOODDataset(None, test_path)
        test_dl = torch.utils.data.DataLoader(test_ds, batch_size=4, shuffle=False)
        acc, ap, roc = inference(resnet, test_dl)

        acc_list[i, j] = acc
        ap_list[i, j] = ap
        roc_list[i, j] = roc

print('Accuracy table:', acc_list)
print('AP table:', ap_list)
print('ROC table:', roc_list)