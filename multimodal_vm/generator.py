import glob

import numpy as np

import utils
import torch
import torch.utils.data as data


class Dataset(data.Dataset):
    def __init__(self, list_a, list_b, transforms, win_low=5, win_high=95, norm=True):
        self.lstA = list_a
        self.lstB = list_b
        self.transforms = transforms
        self.low = win_low
        self.high = win_high
        self.norm = norm

    def __len__(self):
        assert len(self.lstA) == len(self.lstB), 'numA != nmuB'
        return len(self.lstA)

    def __getitem__(self, index):
        x = self.lstA[index]
        x = utils.load_4D(x)
        if self.norm:
            x = utils.irm_min_max_preprocess(x, low_perc=self.low, high_perc=self.high)
        x = self.transforms(x)
        x = np.ascontiguousarray(x)
        x = torch.from_numpy(x)

        y = self.lstB[index]
        y = utils.load_4D(y)
        if self.norm:
            y = utils.irm_min_max_preprocess(y, low_perc=self.low, high_perc=self.high)
        y = self.transforms(y)
        y = np.ascontiguousarray(y)
        y = np.ascontiguousarray(y)
        return x, y

class Dataset_BraTS2022(data.Dataset):
    def __init__(self, list_a, list_b, transforms, win_low=5, win_high=95, norm=True, pre_seg=False, post_seg=False):
        self.lstA = list_a
        self.lstB = list_b
        self.transforms = transforms
        self.low = win_low
        self.high = win_high
        self.norm = norm
        self.pre_seg = pre_seg
        self.post_seg = post_seg

    def __len__(self):
        assert len(self.lstA) == len(self.lstB), 'numA != nmuB'
        return len(self.lstA)

    def __getitem__(self, index):
        x = self.lstA[index]
        x = utils.load_4D(x)
        if self.pre_seg:
            pre_mask = utils.load_4D(self.pre_seg[index])
            x = utils.mask(x, pre_mask)
        x = utils.irm_min_max_preprocess(x, low_perc=self.low, high_perc=self.high)
        x = x[:, 40:-40, 24:-24, :]
        x = np.pad(x, ((0, 0), (0, 0), (0, 0), (3, 2)), 'constant')
        # x = utils.bounding_box(x)
        # x = utils.pad_or_crop_image(x, target_size=(160, 192, 160))
        x = self.transforms(x)
        x = np.ascontiguousarray(x)
        x = torch.from_numpy(x)

        y = self.lstB[index]
        y = utils.load_4D(y)
        if self.post_seg:
            post_seg = utils.load_4D(self.post_seg[index])
            y = utils.mask(y, post_seg)
        y = utils.irm_min_max_preprocess(y, low_perc=self.low, high_perc=self.high)
        y = y[:, 40:-40, 24:-24, :]
        y = np.pad(y, ((0, 0), (0, 0), (0, 0), (3, 2)), 'constant')
        # y = utils.bounding_box(y)
        # y = utils.pad_or_crop_image(y, target_size=(160, 192, 160))
        y = self.transforms(y)
        y = np.ascontiguousarray(y)
        y = torch.from_numpy(y)
        return x, y