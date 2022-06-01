import csv
import glob
import math
import os

import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F

import losses
import network
import utils

model_path = '/home/hpm/multi_reg/multimodal_vm/log/oasis/model/model_0485.pth'
model = network.VmNet(
    inshape=(160, 224, 192), in_chs=2, enc_chs=[16, 32, 32, 32], dec_chs=[32, 32, 32, 32, 32, 16, 16]
)
model.load_state_dict(torch.load(model_path))

model.to('cuda')
model.eval()

imglist1 = sorted(glob.glob('/home/hpm/downloads/OASIS3_procession/multi_reg/t1_seg/*/*N4.nii.gz'))
imglist2 = sorted(glob.glob('/home/hpm/downloads/OASIS3_procession/multi_reg/t2_seg/*/*N4.nii.gz'))
kplist = sorted(glob.glob('/home/hpm/downloads/temp/sift/sub*.txt'))
tre_ori = []
tre = []
for i in range(len(kplist)):
    imgpath1 = imglist1[i]
    imgpath2 = imglist2[i]
    kp = kplist[i]

    cor1 = []
    cor2 = []
    with open(kp, 'r') as f:
        for line in f.readlines():
            point1 = line.split(';')[0]
            x = int(point1.split(',')[0])
            y = int(point1.split(',')[1])
            z = int(point1.split(',')[2])
            cor = [x, y, z]
            cor1.append(cor)
            point2 = line.split(';')[1]
            x = int(point2.split(',')[0])
            y = int(point2.split(',')[1])
            z = int(point2.split(',')[2])
            cor = [x, y, z]
            cor2.append(cor)
            
    tre_ori_this = []
    for kp_num in range(len(cor1)):
        cor_1 = np.array(cor1[kp_num])
        cor_2 = np.array(cor2[kp_num])
        err = np.ones_like(cor_1)
        err = cor_1 - cor_2
        err = np.sum(err * err)
        tre_ori_this.append(np.sqrt(err))
    tre_ori.append(np.median(tre_ori_this))


    img1 = nib.load(imgpath1)
    img2 = nib.load(imgpath2)
    arr1 = img1.get_fdata()
    arr2 = img1.get_fdata()
    arr1 = utils.irm_min_max_preprocess(arr1, 5, 95)
    arr2 = utils.irm_min_max_preprocess(arr2, 5, 95)
    arr1 = torch.from_numpy(arr1).unsqueeze(0).unsqueeze(0).cuda().float()
    arr2 = torch.from_numpy(arr2).unsqueeze(0).unsqueeze(0).cuda().float()
    # print(arr1.shape)
    with torch.no_grad():
        # x = torch.cat([arr1, arr2], dim=1)
        y_pred, flow = model(arr1, arr2, train=False)
    flow_out = flow.detach().cpu().numpy().squeeze()
    # y_pred_out = y_pred.detach().cpu().numpy().squeeze()
    # utils.save_img(y_pred_out, '/home/hpm/downloads/temp/sub-OAS30479_ses-d1266.nii.gz', '/home/hpm/downloads/OASIS3_procession/multi_reg/t1_seg/sub-OAS30476_ses-d0090_T1w/sub-OAS30476_ses-d0090_T1w_N4.nii.gz')
    tre_this = []
    for kp_num in range(len(cor1)):
        cor_1 = cor1[kp_num]
        cor_2 = cor2[kp_num]
        flow_out_this = flow_out[:, cor_1[0], cor_1[1], cor_1[2]]
        new_loc = cor_1 - flow_out_this
        err = new_loc - cor_2
        err = np.sum(err * err)
        tre_this.append(np.sqrt(err))
    tre.append(np.median(tre_this))
    print(i, np.median(tre_ori_this), np.median(tre_this))
print(np.mean(tre_ori))
print(np.mean(tre))
