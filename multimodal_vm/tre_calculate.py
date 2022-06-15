import csv
import glob
import math
import os

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
import torch.utils.data as Data
from torchvision import transforms

import generator
import losses
import network
import trans
import utils
from trans import Constant


def landmarks_csv2arr(csv_file):
    coor_a = []
    with open(csv_file, 'r') as lm:
        csv_reader = csv.reader(lm)
        for row in csv_reader:
            if row[0] == 'Landmark':
                continue
            x = (round(float(row[1])))
            y = (round(float(row[2])))
            z = (round(float(row[3])))
            coor = [x, y + 240, z]
            # print(coor)
            coor_a.append(coor)
    return np.array(coor_a)


def sift_txt2arr(txt_file):
    coor_a = []
    with open(txt_file, 'r') as txt:
        for lines in txt.readlines():
            kp = lines.split(';')
            x1 = int(kp[0].split(',')[0])
            y1 = int(kp[0].split(',')[1])
            z1 = int(kp[0].split(',')[2])
            coor = [x1, y1, z1]
            coor_a.append(coor)
    return np.array(coor_a)


def coor_distance(coor1, coor2):
    return np.sqrt(np.sum(np.power((coor1 - coor2), 2)))


def plot_kp_nii(coor_list, nii_path):
    img = nib.load(nii_path)
    arr = img.get_fdata()
    for i in range(coor_list.shape[0]):
        x = coor_list[i][0]
        y = coor_list[i][1]
        z = coor_list[i][2]
        # print(x, y, z)
        fig = plt.figure(num=1, figsize=(15, 5))
        ax = fig.add_subplot(111)
        ax.imshow(arr[:, :, z], cmap='gray')
        ax.plot(y, x, c='r', marker='x')
        plt.show()


test_dir = './log/20220613/test_result/'
utils.mkdir(test_dir)
dir = '/home/hpm/downloads/BraTS2022/training_some/'
model_path = './log/20220613/model/model_0400.pth'
file_dir = './log/20220613/tre_400.txt'
model = network.VmNet(
    inshape=(160, 192, 160), in_chs=2, enc_chs=[16, 32, 32, 32], dec_chs=[32, 32, 32, 32, 32, 16, 16]
)
model.load_state_dict(torch.load(model_path))

model.to('cuda')
model.eval()

imglist1 = sorted(glob.glob(dir + '*/*_00_????_t1_N4.nii.gz'))
imglist2 = sorted(glob.glob(dir + '*/*_01_????_t1_N4.nii.gz'))

pre_seg_path = sorted(glob.glob(dir + '*/*_00_????_seg.nii.gz'))
post_seg_path = sorted(glob.glob(dir + '*/*_01_????_seg.nii.gz'))

pre_landmarks = sorted(glob.glob(dir + '*/*_00_????_landmarks.csv'))
post_landmarks = sorted(glob.glob(dir + '*/*_01_????_landmarks.csv'))

imglist1 = imglist1[round(len(imglist1) * 0.8): ]
imglist2 = imglist2[round(len(imglist2) * 0.8): ]

test_pre_seg_path = pre_seg_path[round(len(pre_seg_path) * 0.8):]
test_post_seg_path = post_seg_path[round(len(post_seg_path) * 0.8):]

pre_landmarks = pre_landmarks[round(len(pre_landmarks) * 0.8): ]
post_landmarks = post_landmarks[round(len(post_landmarks) * 0.8): ]
# lm_coor = landmarks_csv2arr(landmarks[0])

data_trans = transforms.Compose(
    [
        # trans.Pad3DIfNeeded((192, 180, 180)),
        # trans.CenterCropBySize((160, 256, 256)),
        trans.NumpyType((np.float32, np.float32))
    ]
)

test_set = generator.Dataset_BraTS2022(list_a=imglist1, list_b=imglist2, transforms=data_trans, pre_seg=False, post_seg=False)
test_Loader = Data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)


with open(file_dir, 'w') as f:
    tre_ori = []
    tre = []

    for i, data in enumerate(test_Loader):
        # if i != 11:
            # continue
        pre_lm = landmarks_csv2arr(pre_landmarks[i])
        post_lm = landmarks_csv2arr(post_landmarks[i])
        pre_seg = pre_seg_path[i]
        pre_seg = utils.load_4D(pre_seg)
        pre_seg = pre_seg[np.newaxis, ...]
        pre_seg[pre_seg != 0] = 1

        arr1, arr2 = data
        arr1 = arr1.cuda()
        arr2 = arr2.cuda()
        arr1_ori = arr1.detach().cpu().numpy().squeeze()
        arr1_ori = np.pad(arr1_ori, ((40, 40), (24, 24), (0, 0)), 'constant')
        arr1_ori = arr1_ori[:, :, 3:-2]
        arr2_ori = arr2.detach().cpu().numpy().squeeze()
        arr2_ori = np.pad(arr2_ori, ((40, 40), (24, 24), (0, 0)), 'constant')
        arr2_ori = arr2_ori[:, :, 3:-2]
        # print(arr1.shape)
        with torch.no_grad():
            y_pred, flow = model(arr1, arr2, train=False)
        flow_out = flow.detach().cpu().numpy()
        flow_out = np.pad(flow_out, ((0, 0), (0, 0), (40, 40), (24, 24), (0, 0)), 'constant')
        flow_out = flow_out[:, :, :, :, 3:-2]
        # utils.save_flow(flow_out, test_dir + f'ori_flow_{i}.nii.gz', imglist1[0])

        # blur = trans.GaussianBlur(dim=3, sigma=Constant(10.0))
        # flow_out = blur(flow_out)
        # utils.save_flow(flow_out, test_dir + f'blur_flow_{i}.nii.gz', imglist1[0])

        y_out = y_pred.detach().cpu().numpy().squeeze()
        y_out = np.pad(y_out, ((40, 40), (24, 24), (0, 0)), 'constant')
        y_out = y_out[:, :, 3:-2]

        # utils.save_img(arr1_ori,  test_dir + f'moving_{i}.nii.gz', imglist1[0])
        # utils.save_img(arr2_ori,  test_dir + f'fixed_{i}.nii.gz', imglist1[0])
        # utils.save_flow(flow_out, test_dir + f'flow_{i}.nii.gz', imglist1[0])
        # utils.save_flow(blur_flow_out, test_dir + f'blur_flow_{i}.nii.gz', imglist1[0])
        # utils.save_img(y_out,  test_dir + f'pred_{i}.nii.gz', imglist1[0])
        # print(flow_out.shape)
        
        tre_ori_this = []
        tre_this = []
        for kp_num in range(len(pre_lm)):
            cor_1 = np.array(pre_lm[kp_num])
            cor_2 = np.array(post_lm[kp_num])
            # err = np.ones_like(cor_1)
            pre_err = cor_1 - cor_2
            pre_err = np.sum(pre_err * pre_err)
            tre_ori_this.append(np.sqrt(pre_err))
            flow_out_this = flow_out.squeeze()[:, cor_1[0], cor_1[1], cor_1[2]]
            new_loc = cor_1 - flow_out_this
            post_err = new_loc - cor_2
            post_err = np.sum(post_err * post_err)
            tre_this.append(np.sqrt(post_err))
            # print(kp_num)
            # print('pre:', cor_1)
            # print('post:', cor_2)
            # print('new:', new_loc)
            # print('pre:', np.sqrt(pre_err), 'post:', np.sqrt(post_err))

            # fig = plt.figure(num=1, figsize=(30, 8))
            # ax = fig.add_subplot(131)
            # ax.imshow(arr1_ori[:, :, cor_1[2]], cmap='gray')
            # ax.plot(cor_1[1], cor_1[0], c='r', marker='x')
            # ax = fig.add_subplot(132)
            # ax.imshow(arr2_ori[:, :, cor_2[2]], cmap='gray')
            # ax.plot(cor_2[1], cor_2[0], c='r', marker='x')
            # ax = fig.add_subplot(133)
            # ax.imshow(y_out[:, :, round(new_loc[2])], cmap='gray')
            # ax.plot(round(new_loc[1]), round(new_loc[0]), c='r', marker='x')
            # plt.show()

        tre_ori.append(np.median(tre_ori_this))
        tre.append(np.median(tre_this))
        print(i, 'pre:', np.median(tre_ori_this), 'post:', np.median(tre_this))
        # print(i, 'pre:', np.median(tre_ori_this), 'post:', np.median(tre_this), file=f)
    # print('average')
    print('pre:', np.mean(tre_ori), 'post:', np.mean(tre))
    # # print()
    # print('pre:', np.mean(tre_ori), file=f)
    # print('post:', np.mean(tre), file=f)

