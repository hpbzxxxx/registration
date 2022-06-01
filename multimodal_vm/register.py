import os
import glob
import time

import utils
import generator
import network
import trans
import losses

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as Data
from torchvision import transforms

test_dir = './log/20220525_1/test_result/'
utils.mkdir(test_dir)
s_time = time.time()
# f = open('/home/hpm/downloads/neurite-oasis.v1.0/vm_test/result_400epoch/dice.txt', 'w')
model_path = './log/20220525_1/model/model_0800.pth'
dir = '/home/hpm/downloads/BraTS2022/training_some/'
# a_dir = '/home/hpm/downloads/OASIS3_procession/multi_reg/t1_160_224_192/'
# b_dir = '/home/hpm/downloads/OASIS3_procession/multi_reg/t2_160_224_192/'
# moved_seg_path = './result/moved_seg.nii.gz'

# labels = np.load('F:/xxxx/voxelmorph-dev/data/labels.npz')
# label = [2, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 24, 26, 28, 31, 41, 42, 43, 44, 46, 47, 49, 50, 51, 52, 53, 54, 58, 60, 62, 63, 77]
# label = [2, 3, 4, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 24, 28, 31, 41, 42, 43, 46, 47, 49, 50, 51, 52, 53, 54, 60, 63]
label = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
         31, 32, 33, 34, 35]

# device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

a_path = sorted(glob.glob(dir + '*/*_00_????_t1.nii.gz'))
b_path = sorted(glob.glob(dir + '*/*_01_????_t1.nii.gz'))
# a_path = sorted(glob.glob(a_dir + '*.nii.gz'))
# b_path = sorted(glob.glob(b_dir + '*.nii.gz'))

# load fixed images
# fixed_img_path = '/home/hpm/downloads/OASIS3_procession/multi_reg/t2_seg/sub-OAS30476_ses-d0090_T2w/sub-OAS30476_ses-d0090_T2w_N4.nii.gz'
# fixed_seg = torch.from_numpy(utils.load_4D('/home/hpm/downloads/OASIS3_procession/multi_reg/t2_seg/sub-OAS30476_ses-d0090_T2w/sub-OAS30476_ses-d0090_T2w_fusion.nii.gz')).cuda().unsqueeze(0).float()
# moving_img_path = '/home/hpm/downloads/OASIS3_procession/multi_reg/t1_seg/sub-OAS30476_ses-d0090_T1w/sub-OAS30476_ses-d0090_T1w_N4.nii.gz'
# moving_seg = torch.from_numpy(utils.load_4D('/home/hpm/downloads/OASIS3_procession/multi_reg/t1_seg/sub-OAS30476_ses-d0090_T1w/sub-OAS30476_ses-d0090_T1w_fusion.nii.gz')).cuda().unsqueeze(0).float()
test_path_a = a_path[round(len(a_path) * 0.8):]
test_path_b = b_path[round(len(b_path) * 0.8):]

data_trans = transforms.Compose(
    [
        # trans.Pad3DIfNeeded((192, 180, 180)),
        # trans.CenterCropBySize((160, 256, 256)),
        trans.NumpyType((np.float32, np.float32))
    ]
)

test_set = generator.Dataset_BraTS2022(list_a=test_path_a, list_b=test_path_b, transforms=data_trans)
test_Loader = Data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)

# load and set up model
model = network.VmNet(
    inshape=(160, 192, 160), in_chs=2, enc_chs=[16, 32, 32, 32], dec_chs=[32, 32, 32, 32, 32, 16, 16]
)
model.load_state_dict(torch.load(model_path))

# config = config_affine.get_3DTransMorphAffine_config()
# model = network.TransMorphAffine(config)

model.to(device)
model.eval()

# predict
with torch.no_grad():
    for i, data in enumerate(test_Loader):
        img_a, img_b = data

        img_a = img_a.to(device)
        img_b = img_b.to(device)

        pred_img, flow = model(img_a, img_b, train=False)

        moving = img_a.detach().cpu().numpy().squeeze()
        fixed = img_b.detach().cpu().numpy().squeeze()
        y_pred_out = pred_img.detach().cpu().numpy().squeeze()
        flow_out = flow.detach().cpu().numpy().squeeze()
        utils.save_img(
            y_pred_out, test_dir + f'pred_{i}.nii.gz', test_path_a[0]
        )
        utils.save_flow(
            flow_out, test_dir + f'flow_{i}.nii.gz', test_path_a[0]
        )
        utils.save_img(
            moving, test_dir + f'moving_{i}.nii.gz', test_path_a[0]
        )
        utils.save_img(
            fixed, test_dir + f'fixed_{i}.nii.gz', test_path_a[0]
        )
        print(i)
e_time = time.time()
print(e_time - s_time)
