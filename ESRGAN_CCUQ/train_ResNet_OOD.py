import pandas as pd
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.io as sio
from numpy.linalg import inv
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.feature_selection import SelectFromModel
import os, re, glob
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ['CUDA_VISIBLE_DEVICES'] = '3'    # specify which GPU(s) to be used
import torch


class MixOODDataset(torch.utils.data.Dataset):
    def __init__(self, id_paths, ood_paths, transform=None):
        self.id_paths = id_paths
        self.ood_paths = ood_paths
        self.transform = transform
        id_imgs, ood_imgs = [], []
        for id_p in self.id_paths:
            y0_list = glob.glob(os.path.join(id_p, '*_0.png'))
            id_imgs.extend(y0_list)
        for ood_p in self.ood_paths:
            y0_list = glob.glob(os.path.join(ood_p, '*_0.png'))
            ood_imgs.extend(y0_list)
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

def main():
    DATA_PATH = 'C:\\Jacky\\datasets'
    TRAIN_MODEL = 'a'  # 'a' or 'm' or 'f'
    if TRAIN_MODEL == 'a':
        TRAIN_SCENES = ['anime_dataset']
        TEST_SCENES = ['face_dataset']
    elif TRAIN_MODEL == 'm':
        TRAIN_SCENES = ['micro_dataset']
        TEST_SCENES = ['face_dataset']
    elif TRAIN_MODEL == 'f':
        TRAIN_SCENES = ['face_dataset']
        TEST_SCENES = ['micro_dataset']

    model_name = 'ResNet50_%s' % TRAIN_MODEL
    # traverse all train scenes and get the paths of inlier and outlier
    train_paths = []
    for scene in TRAIN_SCENES:
        train_paths.extend(glob.glob(os.path.join(DATA_PATH, scene, '**')))

    # Classify the paths into inlier and outlier
    train_id_paths, train_ood_paths = [], []

    for p in train_paths:
        p_name = os.path.basename(p)  # get the name of the path

        if 'id' in p_name:
            train_id_paths.append(p)
            print('Training ID path:', p)
        elif 'noised' in p_name:
            train_ood_paths.append(p)
            print('Training OOD path:', p)

    # traverse all test scenes and get the paths of inlier and outlier
    test_paths = []
    for scene in TEST_SCENES:
        test_paths.extend(glob.glob(os.path.join(DATA_PATH, scene, '**')))

    # Classify the paths into inlier and outlier
    test_id_paths = train_id_paths
    test_ood_paths = []

    for p in test_paths:
        p_name = os.path.basename(p)  # get the name of the path

        if TRAIN_MODEL == 'a':
            if 'anime_cycle' in p_name:
                test_ood_paths.append(p)
                print('Testing OOD path:', p)
        elif TRAIN_MODEL == 'm':
            if 'micro_cycle' in p_name:
                test_ood_paths.append(p)
                print('Testing OOD path:', p)
        elif TRAIN_MODEL == 'f':
            if 'face_cycle' in p_name:
                test_ood_paths.append(p)
                print('Testing OOD path:', p)


    # summary data paths
    print('Total ID train paths:', len(train_id_paths))
    print('Total OOD train paths:', len(train_ood_paths))
    print('Total ID test paths:', len(test_id_paths))
    print('Total OOD test paths:', len(test_ood_paths))

    # saving path
    os.makedirs('./ResNet_OOD_ckpt', exist_ok=True)


    # Dataset and DataLoader
    train_dataset = MixOODDataset(train_id_paths, train_ood_paths)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
    valid_dataset = MixOODDataset(test_id_paths, test_ood_paths)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=4, shuffle=False, num_workers=4)

    # Model
    resnet = torch.hub.load('pytorch/vision:v0.14.1', 'resnet50', pretrained=False).cuda()

    # Loss and optimizer
    ce_loss = torch.nn.CrossEntropyLoss()
    opt = torch.optim.Adam(resnet.parameters(), lr=1e-4)


    # Training
    best_valid_loss = 1e10
    for e in range(1000):
        resnet.train()
        train_loss = 0
        for i, (x, y) in enumerate(train_dataloader):
            x = x.cuda()
            y = y.cuda()
            y_pred = resnet(x)
            loss = ce_loss(y_pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item()
        print('Epoch: {}, Train loss: {}'.format(e, train_loss / (i+1)))

        if e % 10 == 0:
            with torch.no_grad():
                resnet.eval()
                valid_loss = 0
                for i, (x, y) in enumerate(valid_dataloader):
                    if i >= 10:
                        break
                    x = x.cuda()
                    y = y.cuda()
                    y_pred = resnet(x)
                    loss = ce_loss(y_pred, y)
                    valid_loss += loss.item()
                print('Epoch: {}, Valid loss: {}'.format(e, valid_loss / (i+1)))

            if valid_loss < best_valid_loss:
                torch.save(resnet.state_dict(), './ResNet_OOD_ckpt/%s_best.pth'%model_name)
                best_valid_loss = valid_loss


if __name__ == '__main__':
    main()