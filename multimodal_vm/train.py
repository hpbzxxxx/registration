import glob
import os
import time
from matplotlib.pyplot import setp

import numpy as np
import torch
import torch.utils.data as Data
from torchvision import transforms

import generator
import network
import trans
import utils
import losses

# make dir for log
model_dir = './log/20220601/model/'
loss_dir = './log/20220601/loss/'
val_result = './log/20220601/val_result/'
conf = './log/20220601/config.txt'
utils.mkdir(model_dir)
utils.mkdir(loss_dir)
utils.mkdir(val_result)
with open(conf, 'w') as cfg:
    print('dataset: brats', file=cfg)
    print('model: vm', file=cfg)
    print('loss: bi-MSE+0.01*smooth', file=cfg)
    print('remark: no seg mask, pre_t1-post_t1', file=cfg)

# hyperparameter
epochs = 1500
# iteration_per_epoch = 100
continue_train = False
start_epoch = 0
alpha = 0.01 # 0.01 for mse, 1 for lncc

# enable cuda
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# a_dir = '/home/hpm/downloads/OASIS3_procession/multi_reg/t1_160_224_192/'
# b_dir = '/home/hpm/downloads/OASIS3_procession/multi_reg/t2_160_224_192/'
dir = '/home/hpm/downloads/BraTS2022/training_some/'
learning_rate = 1e-4

# a_path = sorted(glob.glob(a_dir + '*.nii.gz'))
# b_path = sorted(glob.glob(b_dir + '*.nii.gz'))
a_path = sorted(glob.glob(dir + '*/*_00_????_t1.nii.gz'))
b_path = sorted(glob.glob(dir + '*/*_01_????_t1.nii.gz'))
pre_seg_path = sorted(glob.glob(dir + '*/*_00_????_seg.nii.gz'))
post_seg_path = sorted(glob.glob(dir + '*/*_01_????_seg.nii.gz'))

# train dataset
train_path_a = a_path[: round(len(a_path) * 0.7)]
train_path_b = b_path[: round(len(b_path) * 0.7)]
train_pre_seg_path = pre_seg_path[: round(len(pre_seg_path) * 0.7)]
train_post_seg_path = post_seg_path[: round(len(post_seg_path) * 0.7)]
# train_path_a = a_path[: 10]
# train_path_b = b_path[: 10]
# utils.shift_list(train_path_a, 10)

# val dataset
val_path_a = a_path[round(len(a_path) * 0.7) : round(len(a_path) * 0.8)]
val_path_b = b_path[round(len(b_path) * 0.7) : round(len(b_path) * 0.8)]
val_pre_seg_path = pre_seg_path[round(len(pre_seg_path) * 0.7) : round(len(pre_seg_path) * 0.8)]
val_post_seg_path = post_seg_path[round(len(post_seg_path) * 0.7) : round(len(post_seg_path) * 0.8)]
# val_path_a = a_path[: 10]
# val_path_b = b_path[: 10]
# utils.shift_list(val_path_a, 10)

# augmentation
data_trans = transforms.Compose(
    [
        # trans.Pad3DIfNeeded((192, 180, 180)),
        # trans.CenterCropBySize((160, 256, 256)),
        trans.NumpyType((np.float32, np.float32))
    ]
)

train_set = generator.Dataset_BraTS2022(list_a=train_path_a, list_b=train_path_b, transforms=data_trans, pre_seg=train_pre_seg_path, post_seg=train_post_seg_path)
train_Loader = Data.DataLoader(train_set, batch_size=1, shuffle=True, num_workers=0)

val_set = generator.Dataset_BraTS2022(list_a=val_path_a, list_b=val_path_b, transforms=data_trans, pre_seg=val_pre_seg_path, post_seg=val_post_seg_path)
val_Loader = Data.DataLoader(val_set, batch_size=1, shuffle=False, num_workers=0)

# create model
model = network.VmNet(
    inshape=(160, 192, 160), in_chs=2, enc_chs=[16, 32, 32, 32], dec_chs=[32, 32, 32, 32, 32, 16, 16], bi=True
)

model.to(device)
model.train()

# optimizer and losses
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
sim_loss = losses.MSE_loss
# reg_loss = losses.smooth_loss()

# if load model
if continue_train:
    start_epoch = 0
    checkpoint_path = './log/20220601/model/checkpoint_0060.pth'
    model.load_state_dict(torch.load(checkpoint_path)['model'])
    optimizer.load_state_dict(torch.load(checkpoint_path)['optimizer'])
    start_epoch = torch.load(checkpoint_path)['epoch']
    print('load model' + checkpoint_path)

# train epoch
for epoch in range(start_epoch, epochs + 1):

    epoch_total_loss = []
    epoch_sim_loss = []
    epoch_reg_loss = []
    # epoch_step_time = []
    epoch_time_s = time.time()

    with open(loss_dir + 'step_loss.txt', 'a') as steploss:
        for i, data in enumerate(train_Loader):
            # step_start_time = time.time()

            img_a, img_b = data
            img_a = img_a.to(device)
            img_b = img_b.to(device)

            y_pred, y_pred_neg, flow = model(img_a, img_b)

            ################
            # if epoch % 2 == 0 and i % 20 == 0:
            #     t1_img = img_a.detach().cpu().numpy().squeeze()
            #     t2_img = img_b.detach().cpu().numpy().squeeze()
            #     # fea_t1 = fea_a.detach().cpu().numpy().squeeze()
            #     # fea_t2 = fea_b.detach().cpu().numpy().squeeze()
            #     pred = y_pred.detach().cpu().numpy().squeeze()
            #     t = flow.detach().cpu().numpy().squeeze()
            #     utils.save_img(t1_img, f'./train_result/t1_{epoch}_{i}.nii.gz', train_path_a[0])
            #     utils.save_img(t2_img, f'./train_result/t2_{epoch}_{i}.nii.gz', train_path_a[0])
            #     # utils.save_img(fea_t1, f'./train_result/fea_t1_{epoch}_{i}.nii.gz', train_path_a[0])
            #     # utils.save_img(fea_t2, f'./train_result/fea_t2_{epoch}_{i}.nii.gz', train_path_a[0])
            #     utils.save_img(pred, f'./train_result/pred_{epoch}_{i}.nii.gz', train_path_a[0])
            #     utils.save_flow(t, f'./train_result/flow_{epoch}_{i}.nii.gz', train_path_a[0])

            # loss
            sim = 0.5 * sim_loss(y_pred, img_b) + 0.5 * sim_loss(y_pred_neg, img_a)
            # sim = torch.clamp(sim, 0.0, 0.0224)
            reg = losses.smooth_loss(flow)
            loss = sim + alpha * reg

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # step_end_time = time.time()
            # step_time = step_end_time - step_start_time
            epoch_total_loss.append(loss.item())
            epoch_sim_loss.append(sim.item())
            epoch_reg_loss.append(reg.item())
            # epoch_step_time.append(step_time)
            print(
                f'epoch: {epoch}, loss: {loss.item():.4e}, sim: {sim.item():.4e}, reg: {reg.item():.4e}',
                file=steploss
            )

    # scheduler.step(np.mean(epoch_loss))
    epoch_time_e = time.time()
    epoch_time = epoch_time_e - epoch_time_s
    print(
        f'epoch: {epoch}, loss: {np.mean(epoch_total_loss):.4e}, sim: {np.mean(epoch_sim_loss):.4e}, reg: {np.mean(epoch_reg_loss):.4e}, time: {epoch_time:.2e}'
    )
    with open(loss_dir + 'epoch_loss.txt', 'a') as epochloss:
        print(
            f'epoch: {epoch}, loss: {np.mean(epoch_total_loss):.4e}, sim: {np.mean(epoch_sim_loss):.4e}, reg: {np.mean(epoch_reg_loss):.4e}, time: {epoch_time:.2e}',
            file=epochloss
        )

    # val
    with torch.no_grad():
        model.eval()
        val_total_loss = []
        val_sim_loss = []
        val_reg_loss = []

        with open(loss_dir + 'val_loss.txt', 'a') as valloss:

            for i, val_data in enumerate(val_Loader):
                img_a, img_b = val_data
                img_a = img_a.to(device)
                img_b = img_b.to(device)

                y_pred, y_pred_neg, flow = model(img_a, img_b, train=False)

                if epoch % 10 == 2 and i % 5 == 0:
                    moving = img_a.detach().cpu().numpy().squeeze()
                    fixed = img_b.detach().cpu().numpy().squeeze()
                    y_pred_out = y_pred.detach().cpu().numpy().squeeze()
                    y_pred_neg_out = y_pred_neg.detach().cpu().numpy().squeeze()
                    flow_out = flow.detach().cpu().numpy().squeeze()
                    utils.save_img(
                        y_pred_out, f'{val_result}pred{epoch}_{i}.nii.gz', val_path_a[i]
                    )
                    utils.save_flow(
                        flow_out, f'{val_result}flow{epoch}_{i}.nii.gz', val_path_a[i]
                    )
                    utils.save_img(
                        moving, f'{val_result}moving_{i}.nii.gz', val_path_a[i]
                    )
                    utils.save_img(
                        fixed, f'{val_result}fixed_{i}.nii.gz', val_path_a[i]
                    )
                    utils.save_img(
                        y_pred_neg_out, f'{val_result}pred_neg{epoch}_{i}.nii.gz', val_path_a[i]
                    )

                sim = 0.5 * sim_loss(y_pred, img_b) + 0.5 * sim_loss(y_pred_neg, img_a)
                # sim = torch.clamp(sim, 0.0, 0.0224)
                reg = losses.smooth_loss(flow)
                loss = sim + alpha * reg

                val_total_loss.append(loss.item())
                val_sim_loss.append(sim.item())
                val_reg_loss.append(reg.item())
            print(f'epoch: {epoch}, val_loss: {np.mean(val_total_loss):.4e}, sim: {np.mean(val_sim_loss):.4e}, reg: {np.mean(val_reg_loss):.4e}')

            print(f'epoch: {epoch}, loss: {np.mean(val_total_loss):.4e}, sim: {np.mean(val_sim_loss):.4e}, reg: {np.mean(val_reg_loss):.4e}', file=valloss)

    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    if epoch % 20 == 0:
        torch.save(checkpoint, os.path.join(model_dir, 'checkpoint_%04d.pth' % epoch))
    torch.save(model.state_dict(), os.path.join(model_dir, 'model_%04d.pth' % epoch))
