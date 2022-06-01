import os
import math
import random
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F


def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def read_sub_list(filename):
    with open(filename, 'r') as file:
        content = file.readlines()
    sub_path = [x.strip() for x in content if x.strip()]
    return sub_path


def load_4D(imgpath):
    img = nib.load(imgpath)
    arr = img.get_fdata()
    arr = arr.squeeze()
    arr = arr[np.newaxis, ...]  # (1, 160, 192, 224)
    return arr


def save_img(arr, imgpath, ref):
    ref_img = nib.load(ref)
    img = nib.Nifti1Image(arr, ref_img.affine, ref_img.header)
    nib.save(img, imgpath)
    # import SimpleITK as sitk
    # img = sitk.GetImageFromArray(arr)
    # sitk.WriteImage(img, imgpath)


def save_flow(arr, imgpath, ref):
    ref_img = nib.load(ref)
    arr = arr.transpose([1, 2, 3, 0])
    img = nib.Nifti1Image(arr, ref_img.affine, ref_img.header)
    nib.save(img, imgpath)
    # import SimpleITK as sitk
    # img = sitk.GetImageFromArray(arr)
    # sitk.WriteImage(img, imgpath)


def imgnorm_maxmin(img):
    max_v = np.max(img)
    min_v = np.min(img)
    norm_img = (img - min_v) / (max_v - min_v)
    return norm_img


def imgnorm_threshold(img, threshold):
    norm_img = (img - threshold[0]) / (threshold[1] - img)
    norm_img[norm_img > 1] = 1
    norm_img[norm_img < 0] = 0
    return norm_img


def displacement(src, s1, s2, s3, r1, r2, r3, t1, t2, t3, mode='bilinear'):
    scale = torch.tensor([[s1, 0, 0],
                          [0, s2, 0],
                          [0, 0, s3]]).float()
    trans = torch.tensor([t1, t2, t3]).T.unsqueeze(dim=1).float()
    r_x = torch.tensor([[1, 0, 0],
                        [0, math.cos(r1 / 180 * math.pi), -math.sin(r1 / 180 * math.pi)],
                        [0, math.sin(r1 / 180 * math.pi), math.cos(r1 / 180 * math.pi)]]).float()
    r_y = torch.tensor([[math.cos(r2 / 180 * math.pi), 0, math.sin(r2 / 180 * math.pi)],
                        [0, 1, 0],
                        [-math.sin(r2 / 180 * math.pi), 0, math.cos(r2 / 180 * math.pi)]]).float()
    r_z = torch.tensor([[math.cos(r3 / 180 * math.pi), -math.sin(r3 / 180 * math.pi), 0],
                        [math.sin(r3 / 180 * math.pi), math.cos(r3 / 180 * math.pi), 0],
                        [0, 0, 1]]).float()
    rot = r_x @ r_y @ r_z
    r = scale @ rot
    x = src
    theta = torch.cat((r, trans), dim=1)
    theta = theta.unsqueeze(dim=0).cuda()
    grid = F.affine_grid(theta, x.shape, align_corners=True)
    moved = F.grid_sample(x, grid, align_corners=True, mode=mode)
    return theta, moved


def inv_aff(mat):
    """
    3d
    :param mat:
    :return:
    """
    whole_mat = torch.cat((mat, torch.tensor([0, 0, 0, 1]).reshape(1, 1, 4).cuda()), dim=1)
    inv_mat = whole_mat.inverse()[:, :3, :]
    return inv_mat


def irm_min_max_preprocess(image, low_perc=1, high_perc=99):
    """Main pre-processing function used for the challenge (seems to work the best).

    Remove outliers voxels first, then min-max scale.

    Warnings
    --------
    This will not do it channel wise!!
    """

    non_zeros = image > 0
    low, high = np.percentile(image[non_zeros], [low_perc, high_perc])
    image = np.clip(image, low, high)
    image = imgnorm_maxmin(image)
    return image


def shift_list(list, shift, direction='r'):
    if direction == 'r':
        for i in range(shift):
            list.insert(0, list.pop())
    elif direction == 'l':
        for i in range(shift):
            list.insert(len(list), list[0])
            list.remove(list[0])
    else:
        print('direction not correct!')
        exit()


def bounding_box(img_arr):
    x_index, y_index, z_index = np.nonzero(np.sum(img_arr, 0))
    x_min, y_min, z_min = [max(0, int(np.min(cor) - 1)) for cor in (x_index, y_index, z_index)]
    x_max, y_max, z_max = [min(int(np.max(cor) + 1), int(img_arr.shape[i + 1])) for i, cor in enumerate([x_index, y_index, z_index])]
    img_arr = img_arr[:, x_min:x_max, y_min:y_max, z_min:z_max]
    return img_arr


def pad_or_crop_image(image, seg=None, target_size=(128, 144, 144)):
    c, z, y, x = image.shape
    z_slice, y_slice, x_slice = [get_crop_slice(target, dim) for target, dim in zip(target_size, (z, y, x))]
    image = image[:, z_slice, y_slice, x_slice]
    if seg is not None:
        seg = seg[:, z_slice, y_slice, x_slice]
    todos = [get_left_right_idx_should_pad(size, dim) for size, dim in zip(target_size, [z, y, x])]
    padlist = [(0, 0)]  # channel dim
    for to_pad in todos:
        if to_pad[0]:
            padlist.append((to_pad[1], to_pad[2]))
        else:
            padlist.append((0, 0))
    image = np.pad(image, padlist)
    if seg is not None:
        seg = np.pad(seg, padlist)
        return image, seg
    return image


def get_left_right_idx_should_pad(target_size, dim):
    if dim >= target_size:
        return [False]
    elif dim < target_size:
        pad_extent = target_size - dim
        left = random.randint(0, pad_extent)
        right = pad_extent - left
        return True, left, right


def get_crop_slice(target_size, dim):
    if dim > target_size:
        crop_extent = dim - target_size
        left = random.randint(0, crop_extent)
        right = crop_extent - left
        return slice(left, dim - right)
    elif dim <= target_size:
        return slice(0, dim)


def mask(src, mask):
    mask[mask != 0] = 1
    mask = np.logical_not(mask)
    new_img = src * mask
    return new_img
    