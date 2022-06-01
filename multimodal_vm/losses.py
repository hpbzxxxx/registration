import os
import glob
import math
import sklearn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class LNCC_loss:
    """
    Local (over window) normalized cross correlation loss.
    """

    def __init__(self, win=None):
        self.win = win

    def loss(self, y_true, y_pred):

        Ii = y_true
        Ji = y_pred

        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(Ii.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [9] * ndims if self.win is None else self.win

        # compute filters
        sum_filt = torch.ones([1, 1, *win]).to("cuda")

        pad_no = math.floor(win[0] / 2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        I_sum = conv_fn(Ii, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(Ji, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)

        return -torch.mean(cc)



def GNCC_loss(y_true, y_pred):
    x = y_true
    y = y_pred
    dim = len(x.shape) - 2

    x2 = x * x
    y2 = y * y
    xy = x * y

    E_x = torch.mean(x)
    E_y = torch.mean(y)
    E_x2 = torch.mean(x2)
    E_y2 = torch.mean(y2)
    E_xy = torch.mean(xy)

    cov = E_xy - E_x * E_y
    var_x = E_x2 - E_x * E_x
    var_y = E_y2 - E_y * E_y

    # cc = cov * cov / (var_x * var_y + 1e-5)
    cc = cov / (var_x.sqrt() * var_y.sqrt() + 1e-5)
    return -torch.mean(cc)


def MSE_loss(y_true, y_pred):
    return torch.mean((y_true - y_pred) ** 2)


def smooth_loss(flow):
    dx = torch.abs(flow[:, :, 1:, :, :] - flow[:, :, :-1, :, :])
    dy = torch.abs(flow[:, :, :, 1:, :] - flow[:, :, :, :-1, :])
    dz = torch.abs(flow[:, :, :, :, 1:] - flow[:, :, :, :, :-1])

    d = torch.mean(dx * dx) + torch.mean(dy * dy) + torch.mean(dz * dz)
    grad = d / 3.0

    grad *= 2
    return grad


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


# def NMI(x, y):
#     x = x.flatten()
#     y = y.flatten()
#     size = x.shape[-1]
#     px = np.histogram(x, 500)[0] / size
#     py = np.histogram(y, 500)[0] / size
#     hx = - np.sum(px * np.log(px + 1e-8))
#     hy = - np.sum(py * np.log(py + 1e-8))
#
#     joint = torch.cat([x.unsqueeze(dim=1), y.unsqueeze(dim=1)], dim=1)
#     hxy = np.histogramdd(joint, bins=500, range=[[0, 1], [0, 1]])[0]
#     hxy /= (1.0 * size)
#     hxy = - np.sum(hxy * np.log(hxy + 1e-8))
#
#     r = (hx + hy) / hxy / 2
#     return r


# def marginalPdf(values, sigma=0.4, num_bins=256, epsilon=1e-10):
#     bins = torch.linspace(0, 255, num_bins).float().to('cuda')
#     residuals = values - bins.unsqueeze(0).unsqueeze(0)
#     kernel_values = torch.exp(-0.5 * (residuals / sigma).pow(2))

#     pdf = torch.mean(kernel_values, dim=1)
#     normalization = torch.sum(pdf, dim=1).unsqueeze(1) + epsilon
#     pdf = pdf / normalization

#     return pdf, kernel_values


# def jointPdf(kernel_values1, kernel_values2):
#     joint_kernel_values = torch.matmul(kernel_values1.transpose(1, 2), kernel_values2)
#     normalization = torch.sum(joint_kernel_values, dim=(1, 2)).view(-1, 1, 1) + 1e-10
#     pdf = joint_kernel_values / normalization

#     return pdf


# def getMutualInformation(input1, input2, normalize=True, epsilon=1e-10):
#     """

#     :param epsilon:
#     :param normalize:
#     :param input1: B, C, D, H, W
#     :param input2: B, C, D, H, W
#     :return: scalar
#     """

#     epsilon = 1e-10

#     # Torch tensors for images between (0, 1)
#     input1 = input1 * 255
#     input2 = input2 * 255

#     B, C, D, H, W = input1.shape
#     assert (input1.shape == input2.shape)

#     x1 = input1.view(B, D * H * W, C)
#     x2 = input2.view(B, D * H * W, C)

#     pdf_x1, kernel_values1 = marginalPdf(x1)
#     pdf_x2, kernel_values2 = marginalPdf(x2)
#     pdf_x1x2 = jointPdf(kernel_values1, kernel_values2)

#     H_x1 = -torch.sum(pdf_x1 * torch.log2(pdf_x1 + epsilon), dim=1)
#     H_x2 = -torch.sum(pdf_x2 * torch.log2(pdf_x2 + epsilon), dim=1)
#     H_x1x2 = -torch.sum(pdf_x1x2 * torch.log2(pdf_x1x2 + epsilon), dim=(1, 2))

#     mutual_information = H_x1 + H_x2 - H_x1x2

#     if normalize:
#         mutual_information = 2 * mutual_information / (H_x1 + H_x2)

#     return mutual_information

class MIND_loss(torch.nn.Module):
    def __init__(self, win=None):
        super(MIND_loss, self).__init__()
        self.win = win

    def pdist_squared(self, x):
        xx = (x ** 2).sum(dim=1).unsqueeze(2)
        yy = xx.permute(0, 2, 1)
        dist = xx + yy - 2.0 * torch.bmm(x.permute(0, 2, 1), x)
        dist[dist != dist] = 0
        dist = torch.clamp(dist, 0.0, np.inf)
        return dist

    def MINDSSC(self, img, radius=2, dilation=2):
        # see http://mpheinrich.de/pub/miccai2013_943_mheinrich.pdf for details on the MIND-SSC descriptor

        # kernel size
        kernel_size = radius * 2 + 1

        # define start and end locations for self-similarity pattern
        six_neighbourhood = torch.Tensor([[0, 1, 1],
                                          [1, 1, 0],
                                          [1, 0, 1],
                                          [1, 1, 2],
                                          [2, 1, 1],
                                          [1, 2, 1]]).long()

        # squared distances
        dist = self.pdist_squared(six_neighbourhood.t().unsqueeze(0)).squeeze(0)

        # define comparison mask
        x, y = torch.meshgrid(torch.arange(6), torch.arange(6))
        mask = ((x > y).view(-1) & (dist == 2).view(-1))

        # build kernel
        idx_shift1 = six_neighbourhood.unsqueeze(1).repeat(1, 6, 1).view(-1, 3)[mask, :]
        idx_shift2 = six_neighbourhood.unsqueeze(0).repeat(6, 1, 1).view(-1, 3)[mask, :]
        mshift1 = torch.zeros(12, 1, 3, 3, 3).cuda()
        mshift1.view(-1)[torch.arange(12) * 27 + idx_shift1[:, 0] * 9 + idx_shift1[:, 1] * 3 + idx_shift1[:, 2]] = 1
        mshift2 = torch.zeros(12, 1, 3, 3, 3).cuda()
        mshift2.view(-1)[torch.arange(12) * 27 + idx_shift2[:, 0] * 9 + idx_shift2[:, 1] * 3 + idx_shift2[:, 2]] = 1
        rpad1 = nn.ReplicationPad3d(dilation)
        rpad2 = nn.ReplicationPad3d(radius)

        # compute patch-ssd
        ssd = F.avg_pool3d(rpad2(
            (F.conv3d(rpad1(img), mshift1, dilation=dilation) - F.conv3d(rpad1(img), mshift2, dilation=dilation)) ** 2),
                           kernel_size, stride=1)

        # MIND equation
        mind = ssd - torch.min(ssd, 1, keepdim=True)[0]
        mind_var = torch.mean(mind, 1, keepdim=True)
        mind_var = torch.clamp(mind_var, (mind_var.mean() * 0.001).item(), (mind_var.mean() * 1000).item())
        mind /= mind_var
        mind = torch.exp(-mind)

        # permute to have same ordering as C++ code
        mind = mind[:, torch.Tensor([6, 8, 1, 11, 2, 10, 0, 7, 9, 4, 5, 3]).long(), :, :, :]

        return mind

    def forward(self, y_pred, y_true):
        return torch.mean((self.MINDSSC(y_pred) - self.MINDSSC(y_true)) ** 2)


class MutualInformation(torch.nn.Module):
    """
    Mutual Information
    """
    def __init__(self, sigma_ratio=1, minval=0., maxval=1., num_bin=32):
        super(MutualInformation, self).__init__()

        """Create bin centers"""
        bin_centers = np.linspace(minval, maxval, num=num_bin)
        vol_bin_centers = Variable(torch.linspace(minval, maxval, num_bin), requires_grad=False).cuda()
        num_bins = len(bin_centers)

        """Sigma for Gaussian approx."""
        sigma = np.mean(np.diff(bin_centers)) * sigma_ratio

        self.preterm = 1 / (2 * sigma**2)
        self.bin_centers = bin_centers
        self.max_clip = maxval
        self.num_bins = num_bins
        self.vol_bin_centers = vol_bin_centers

    def mi(self, y_true, y_pred):
        y_pred = torch.clamp(y_pred, 0., self.max_clip)
        y_true = torch.clamp(y_true, 0, self.max_clip)

        y_true = y_true.reshape(y_true.shape[0], -1)
        y_true = torch.unsqueeze(y_true, 2)
        y_pred = y_pred.reshape(y_pred.shape[0], -1)
        y_pred = torch.unsqueeze(y_pred, 2)

        nb_voxels = y_pred.shape[1] # total num of voxels

        """Reshape bin centers"""
        o = [1, 1, np.prod(self.vol_bin_centers.shape)]
        vbc = torch.reshape(self.vol_bin_centers, o).cuda()

        """compute image terms by approx. Gaussian dist."""
        I_a = torch.exp(- self.preterm * torch.square(y_true - vbc))
        I_a = I_a / torch.sum(I_a, dim=-1, keepdim=True)

        I_b = torch.exp(- self.preterm * torch.square(y_pred - vbc))
        I_b = I_b / torch.sum(I_b, dim=-1, keepdim=True)

        # compute probabilities
        pab = torch.bmm(I_a.permute(0, 2, 1), I_b)
        pab = pab/nb_voxels
        pa = torch.mean(I_a, dim=1, keepdim=True)
        pb = torch.mean(I_b, dim=1, keepdim=True)

        papb = torch.bmm(pa.permute(0, 2, 1), pb) + 1e-6
        mi = torch.sum(torch.sum(pab * torch.log(pab / papb + 1e-6), dim=1), dim=1)
        return mi.mean() #average across batch

    def forward(self, y_true, y_pred):
        return -self.mi(y_true, y_pred)

class NormalizedMutualInformation(torch.nn.Module):
    """
    Mutual Information
    """
    def __init__(self, sigma_ratio=1, minval=0., maxval=1., num_bin=32):
        super(NormalizedMutualInformation, self).__init__()

        """Create bin centers"""
        bin_centers = np.linspace(minval, maxval, num=num_bin)
        vol_bin_centers = Variable(torch.linspace(minval, maxval, num_bin), requires_grad=False).cuda()
        num_bins = len(bin_centers)

        """Sigma for Gaussian approx."""
        sigma = np.mean(np.diff(bin_centers)) * sigma_ratio

        self.preterm = 1 / (2 * sigma**2)
        self.bin_centers = bin_centers
        self.max_clip = maxval
        self.num_bins = num_bins
        self.vol_bin_centers = vol_bin_centers

    def mi(self, y_true, y_pred):
        y_pred = torch.clamp(y_pred, 0., self.max_clip)
        y_true = torch.clamp(y_true, 0, self.max_clip)

        y_true = y_true.reshape(y_true.shape[0], -1)
        y_true = torch.unsqueeze(y_true, 2)
        y_pred = y_pred.reshape(y_pred.shape[0], -1)
        y_pred = torch.unsqueeze(y_pred, 2)

        nb_voxels = y_pred.shape[1] # total num of voxels

        """Reshape bin centers"""
        o = [1, 1, np.prod(self.vol_bin_centers.shape)]
        vbc = torch.reshape(self.vol_bin_centers, o).cuda()

        """compute image terms by approx. Gaussian dist."""
        I_a = torch.exp(- self.preterm * torch.square(y_true - vbc))
        I_a = I_a / torch.sum(I_a, dim=-1, keepdim=True)

        I_b = torch.exp(- self.preterm * torch.square(y_pred - vbc))
        I_b = I_b / torch.sum(I_b, dim=-1, keepdim=True)

        # compute probabilities
        pab = torch.bmm(I_a.permute(0, 2, 1), I_b)
        pab = pab/nb_voxels
        pa = torch.mean(I_a, dim=1, keepdim=True)
        pb = torch.mean(I_b, dim=1, keepdim=True)

        papb = torch.bmm(pa.permute(0, 2, 1), pb) + 1e-6
        mi = torch.sum(torch.sum(pab * torch.log(pab / papb + 1e-6), dim=1), dim=1)
        ha = torch.sum(torch.sum(pa * torch.log(pa + 1e-6), dim=1))
        hb = torch.sum(torch.sum(pb * torch.log(pb + 1e-6), dim=1))
        nmi = 2 * mi / (ha + hb)
        return nmi.mean() #average across batch

    def forward(self, y_true, y_pred):
        return -self.mi(y_true, y_pred)


def dice(array1, array2, labels=None, include_zero=False):
    """
    Computes the dice overlap between two arrays for a given set of integer labels.

    Parameters:
        array1: Input array 1.
        array2: Input array 2.
        labels: List of labels to compute dice on. If None, all labels will be used.
        include_zero: Include label 0 in label list. Default is False.
    """
    if labels is None:
        labels = np.concatenate([np.unique(a) for a in [array1, array2]])
        labels = np.sort(np.unique(labels))
    if not include_zero:
        labels = np.delete(labels, np.argwhere(labels == 0))

    dicem = np.zeros(len(labels))
    for idx, label in enumerate(labels):
        top = 2 * np.sum(np.logical_and(array1 == label, array2 == label))
        bottom = np.sum(array1 == label) + np.sum(array2 == label)
        bottom = np.maximum(bottom, np.finfo(float).eps)  # add epsilon
        dicem[idx] = top / bottom
    return dicem