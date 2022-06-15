import os
import math
import csv
import random
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
import SimpleITK as sitk


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
    img = nib.Nifti1Image(arr.squeeze(), ref_img.affine, ref_img.header)
    nib.save(img, imgpath)
    # import SimpleITK as sitk
    # img = sitk.GetImageFromArray(arr)
    # sitk.WriteImage(img, imgpath)


def save_flow(arr, imgpath, ref):
    ref_img = nib.load(ref)
    arr = arr.squeeze().transpose([1, 2, 3, 0])
    img = nib.Nifti1Image(arr.squeeze(), ref_img.affine, ref_img.header)
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


def correct_bias(in_file, out_file, image_type=sitk.sitkFloat64):
    '''
    simpleitk N4
    '''
    input_image = sitk.ReadImage(in_file, image_type)
    output_image_s = sitk.N4BiasFieldCorrection(input_image, input_image > 0)
    sitk.WriteImage(output_image_s, out_file)
    return os.path.abspath(out_file)


def calc_histogram(gray_arr, p_range=None):
    gray_arr = gray_arr.reshape(-1, 1)
    if not p_range:
        p_range = (gray_arr.min(), gray_arr.max())
    hists = np.zeros(p_range[1] - p_range[0] + 1)
    for p in gray_arr:
        if p_range[0] <= p <= p_range[1]:
            hists[p - p_range[0]] += 1
    p_range = np.arange(p_range[0], p_range[1] + 1, 1)
    return hists, p_range


def calc_histogram_cdf(hists):
    hists_cumsum = np.cumsum(hists)
    hists_cdf = hists_cumsum / hists_cumsum[-1]
    # hists_cdf = hists_cdf.astype(int)
    return hists_cdf


def histogram_equalization(img_arr, t_range=None):

    # calculate hists
    hists, p_range = calc_histogram(img_arr, t_range)

    # equalization
    # (m, n) = img_arr.shape
    hists_cdf = calc_histogram_cdf(hists)  # calculate CDF
    hists_cdf = np.concatenate([[0], hists_cdf])
    # arr = np.zeros_like(img_arr)
    arr = hists_cdf[img_arr]  # mapping
    if not t_range:
        t_range = p_range
    arr = arr * t_range[-1]
    return arr.astype(int)


def gradient_diff(y_true, y_pred):
    grad_1_vec, grad_1_abs = grad_img(y_true)
    grad_2_vec, grad_2_abs = grad_img(y_pred)
    grad_diff = grad_1_vec - grad_2_vec
    grad_diff = torch.linalg.norm(grad_diff, axis=-1)
    return torch.mean(grad_diff)


def grad_img(x):
    grad_x = x[:, :, 2:, :, :] - x[:, :, :-2, :, :]
    grad_y = x[:, :, :, 2:, :] - x[:, :, :, :-2, :]
    grad_z = x[:, :, :, :, 2:] - x[:, :, :, :, :-2]
    grad_x = F.pad(grad_x, (0, 0, 0, 0, 1, 1), 'constant')
    grad_y = F.pad(grad_y, (0, 0, 1, 1, 0, 0), 'constant')
    grad_z = F.pad(grad_z, (1, 1, 0, 0, 0, 0), 'constant')
    grad_vec = torch.stack([grad_x, grad_y, grad_z], dim=-1)
    grad_abs = torch.sqrt(grad_x * grad_x + grad_y * grad_y + grad_z * grad_z)
    return grad_vec, grad_abs
