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


class Dataset_BraTS2022_concat(data.Dataset):
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
        x_t1 = self.lstA[index]
        x_t2 = x_t1.replace('t1', 't2')
        x_flair = x_t1.replace('t1', 'flair')
        x_t1ce = x_t1.replace('t1', 't1ce')

        x_t1 = utils.load_4D(x_t1)
        x_t2 = utils.load_4D(x_t2)
        x_flair = utils.load_4D(x_flair)
        x_t1ce = utils.load_4D(x_t1ce)

        x_t1 = utils.irm_min_max_preprocess(x_t1, low_perc=self.low, high_perc=self.high)
        x_t2 = utils.irm_min_max_preprocess(x_t2, low_perc=self.low, high_perc=self.high)
        x_flair = utils.irm_min_max_preprocess(x_flair, low_perc=self.low, high_perc=self.high)
        x_t1ce = utils.irm_min_max_preprocess(x_t1ce, low_perc=self.low, high_perc=self.high)

        x = np.concatenate([x_t1, x_t2, x_flair, x_t1ce], axis=0)
        x = x[:, 40:-40, 24:-24, :]
        x = np.pad(x, ((0, 0), (0, 0), (0, 0), (3, 2)), 'constant')
        x = self.transforms(x)
        x = np.ascontiguousarray(x)
        x = torch.from_numpy(x)

        y_t1 = self.lstB[index]
        y_t2 = y_t1.replace('t1', 't2')
        y_flair = y_t1.replace('t1', 'flair')
        y_t1ce = y_t1.replace('t1', 't1ce')

        y_t1 = utils.load_4D(y_t1)
        y_t2 = utils.load_4D(y_t2)
        y_flair = utils.load_4D(y_flair)
        y_t1ce = utils.load_4D(y_t1ce)

        y_t1 = utils.irm_min_max_preprocess(y_t1, low_perc=self.low, high_perc=self.high)
        y_t2 = utils.irm_min_max_preprocess(y_t2, low_perc=self.low, high_perc=self.high)
        y_flair = utils.irm_min_max_preprocess(y_flair, low_perc=self.low, high_perc=self.high)
        y_t1ce = utils.irm_min_max_preprocess(y_t1ce, low_perc=self.low, high_perc=self.high)

        y = np.concatenate([y_t1, y_t2, y_flair, y_t1ce], axis=0)
        y = y[:, 40:-40, 24:-24, :]
        y = np.pad(y, ((0, 0), (0, 0), (0, 0), (3, 2)), 'constant')
        # y = utils.bounding_box(y)
        # y = utils.pad_or_crop_image(y, target_size=(160, 192, 160))
        y = self.transforms(y)
        y = np.ascontiguousarray(y)
        y = torch.from_numpy(y)
        return x, y