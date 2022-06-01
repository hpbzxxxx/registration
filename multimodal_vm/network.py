import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class VmNet(nn.Module):
    def __init__(self,
                 inshape=None,
                 in_chs=None,
                 enc_chs=None,
                 dec_chs=None,
                 max_pool=2,
                 bi=False):
        super(VmNet, self).__init__()

        dims = len(inshape)
        self.nb_level = len(enc_chs)

        remain_chs = dec_chs[self.nb_level:]
        dec_chs = dec_chs[:self.nb_level]

        # encoder
        self.encoder = nn.ModuleList()
        pre_chs = in_chs
        for level in range(self.nb_level):
            chs = enc_chs[level]
            conv = ConvBlock(dims, pre_chs, chs, 3, 1, 1, 0.2)
            pre_chs = chs
            self.encoder.append(conv)
        MaxPooling = getattr(nn, 'MaxPool%dd' % dims)

        # decoder
        self.decoder = nn.ModuleList()
        cat_chs = np.flip(enc_chs)
        for level in range(self.nb_level):
            chs = dec_chs[level]
            conv = ConvBlock(dims, pre_chs, chs, 3, 1, 1, 0.2)
            pre_chs = chs
            self.decoder.append(conv)
            pre_chs += cat_chs[level]

        self.pooling = MaxPooling(max_pool)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # remaining
        self.remaining = nn.ModuleList()
        for level in range(len(remain_chs)):
            chs = remain_chs[level]
            conv = ConvBlock(dims, pre_chs, chs, 3, 1, 1, 0.2)
            pre_chs = chs
            self.remaining.append(conv)

        self.flow = nn.Conv3d(remain_chs[-1], dims, kernel_size=3, padding=1)

        self.bi = bi

        self.resize = ResizeTransform(1 / 2, dims)
        self.fullsize = ResizeTransform(2, dims)

        downshape = [int(dim / 2) for dim in inshape]
        self.integate = VectorInt(downshape, dims)

        self.transform = SpatialTransformer(inshape)

    def forward(self, src, target, train=True, seg=False):
        x = torch.cat([src, target], dim=1)
        x_concatenation = [x]
        # encoder
        for level, conv in enumerate(self.encoder):
            x = conv(x)
            x_concatenation.append(x)
            x = self.pooling(x)

        # decoder
        for level, conv in enumerate(self.decoder):
            x = conv(x)
            x = self.upsample(x)
            x = torch.cat([x, x_concatenation.pop()], dim=1)

        # remaining
        for conv in self.remaining:
            x = conv(x)

        pos_flow = self.flow(x)

        # integrate for diffeomorphic
        pos_flow = self.resize(pos_flow)
        preint_flow = pos_flow
        if self.bi:
            neg_flow = -preint_flow
        pos_flow = self.integate(pos_flow)
        if self.bi:
            neg_flow = self.integate(neg_flow)
        pos_flow = self.fullsize(pos_flow)
        if self.bi:
            neg_flow = self.fullsize(neg_flow)

        # apply transform
        y_pred = self.transform(src, pos_flow)
        if self.bi:
            y_pred_neg = self.transform(target, neg_flow)
        
        if train:
            if self.bi:
                return y_pred, y_pred_neg, preint_flow
            return y_pred, preint_flow
        else:
            if self.bi:
                return y_pred, y_pred_neg, pos_flow
            return y_pred, pos_flow


class ConvBlock(nn.Module):
    def __init__(self, dims, in_channels, out_channels, kernel_size, stride, padding, negative_slope):
        super().__init__()
        Conv = getattr(nn, 'Conv%dd' % dims)
        self.conv = Conv(in_channels, out_channels, kernel_size, stride, padding)
        self.activate = nn.LeakyReLU(negative_slope)

    def forward(self, x):
        x = self.conv(x)
        x = self.activate(x)
        return x


class SpatialTransformer(nn.Module):
    def __init__(self, shape, mode='bilinear'):
        super(SpatialTransformer, self).__init__()

        self.mode = mode

        vectors = [torch.arange(s) for s in shape]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)
        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        new_locs = flow + self.grid
        shape = flow.shape[2:]

        for i in range(len(shape)):
            new_locs[:, i, ...] = new_locs[:, i, ...] / (shape[i] - 1) * 2 - 1

        if len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return F.grid_sample(src, new_locs, align_corners=True, mode=self.mode)


class VectorInt(nn.Module):
    def __init__(self, shape, step):
        super().__init__()
        self.step = step
        self.transform = SpatialTransformer(shape)

    def forward(self, flow):
        flow = flow * (1 / 2 ** self.step)
        for i in range(self.step):
            flow = flow + self.transform(flow, flow)
        return flow


class ResizeTransform(nn.Module):
    def __init__(self, scale, dims):
        super().__init__()
        self.scale = scale
        if dims == 2:
            self.mode = 'bilinear'
        elif dims ==3:
            self.mode = 'trilinear'

    def forward(self, x):
        if self.scale > 1:
            x = F.interpolate(x, align_corners=True, scale_factor=self.scale, mode=self.mode, recompute_scale_factor=True)
            x = x * self.scale
        elif self.scale < 1:
            x = x * self.scale
            x = F.interpolate(x, align_corners=True, scale_factor=self.scale, mode=self.mode, recompute_scale_factor=True)

        return x