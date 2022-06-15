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

# VOL_SIZE = 21


# def make_gaussian_kernel(sigma):
#     ks = int(sigma * 5)
#     if ks % 2 == 0:
#         ks += 1
#     ts = torch.linspace(-ks // 2, ks // 2 + 1, ks)
#     gauss = torch.exp((-(ts / sigma)**2 / 2))
#     kernel = gauss / gauss.sum()

#     return kernel

    
# def test_3d_gaussian_blur(blur_sigma=2):
#     # Make a test volume
#     vol = torch.randn([VOL_SIZE] * 3) # using something other than zeros
#     vol[VOL_SIZE // 2, VOL_SIZE // 2, VOL_SIZE // 2] = 1

#     # 3D convolution
#     vol_in = vol.reshape(1, 1, *vol.shape)
#     k = make_gaussian_kernel(blur_sigma)
#     k3d = torch.einsum('i,j,k->ijk', k, k, k)
#     k3d = k3d / k3d.sum()
#     vol_3d = F.conv3d(vol_in, k3d.reshape(1, 1, *k3d.shape), stride=1, padding=len(k) // 2)

#     # Separable 1D convolution
#     vol_in = vol[None, None, ...]
#     # k2d = torch.einsum('i,j->ij', k, k)
#     # k2d = k2d / k2d.sum() # not necessary if kernel already sums to zero, check:
#     # print(f'{k2d.sum()=}')
#     k1d = k[None, None, :, None, None]
#     for i in range(3):
#         vol_in = vol_in.permute(0, 1, 4, 2, 3)
#         vol_in = F.conv3d(vol_in, k1d, stride=1, padding=(len(k) // 2, 0, 0))
#     vol_3d_sep = vol_in
#     print((vol_3d- vol_3d_sep).abs().max()) # something ~1e-7
#     print(torch.allclose(vol_3d, vol_3d_sep)) # allclose checks if it is around 1e-8

# img = utils.load_4D('/home/hpm/downloads/OASIS3_procession/multi_reg/t1_160_224_192/sub-OAS30001_ses-d0129_run-01_T1w.nii.gz')
# img = img[np.newaxis, ...]
# # img = img.permute(0, 2, 3, 4, 1)
# img = torch.from_numpy(img).float()
# k = make_gaussian_kernel(7)
# k3d = torch.einsum('i,j,k->ijk', k, k, k)
# k3d = k3d / k3d.sum()
# new_img = F.conv3d(img, k3d.reshape(1, 1, *k3d.shape), stride=1, padding=len(k) // 2)


# new_img = new_img.cpu().numpy().squeeze()
# utils.save_img(new_img, '/home/hpm/downloads/temp/blur1.nii.gz', '/home/hpm/downloads/OASIS3_procession/multi_reg/t1_160_224_192/sub-OAS30001_ses-d0129_run-01_T1w.nii.gz')

# test_3d_gaussian_blur()
# print('*********')



##################################################


# dir = '/home/hpm/downloads/BraTS2022/training_some/'
# imglist = sorted(glob.glob(dir + '*/*t1.nii.gz'))
# for i in range(len(imglist)):
#     img = imglist[i]
#     print(img)

#     # img_hists, img_range = calc_histogram(img, p_range=(1, img.max()))
#     he_img = utils.correct_bias(img, img.replace('.nii.gz', '_N4.nii.gz'))

fixed_dir = '/home/hpm/multi_reg/multimodal_vm/log/20220525_1/val_result/fixed_0.nii.gz'
moving_dir = '/home/hpm/multi_reg/multimodal_vm/log/20220525_1/val_result/moving_0.nii.gz'
pred_dir = '/home/hpm/multi_reg/multimodal_vm/log/20220525_1/val_result/pred742_0.nii.gz'
fixed = nib.load(fixed_dir).get_fdata().reshape([1, 1, 160, 192, 160])
moving = nib.load(moving_dir).get_fdata().reshape([1, 1, 160, 192, 160])
pred = nib.load(pred_dir).get_fdata().reshape([1, 1, 160, 192, 160])

fixed = torch.from_numpy(fixed).cuda().float()
moving = torch.from_numpy(moving).cuda().float()
pred = torch.from_numpy(pred).cuda().float()
sim_loss = losses.LNCC_loss().loss

print(sim_loss(fixed, moving))
print(sim_loss(fixed, pred))
# print(sim_loss(fixed, moving))
