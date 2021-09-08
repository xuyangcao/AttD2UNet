import os
import cv2
import sys
import tqdm
import shutil
import random
import logging
import argparse
import numpy as np
import setproctitle
import matplotlib.pyplot as plt
from skimage.color import label2rgb
plt.switch_backend('agg')
from ast import literal_eval

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid

from dataloaders import make_data_loader
from utils.utils import save_checkpoint, confusion, get_dice
from utils.loss import DiceLoss, dice_loss, FocalTiLoss, FocalDiceLoss, FocalLoss, boundary_loss
from utils.lr_scheduler import LR_Scheduler
from models.attd2unet import AttD2UNet
from models.d2unet import D2UNet

def get_args():
    print('initing args------')
    parser = argparse.ArgumentParser()

    # general config
    parser.add_argument('--gpu', default='0', type=str)
    parser.add_argument('--ngpu', default=1, type=str)
    parser.add_argument('--seed', default=6, type=int)
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--start_epoch', default=1, type=int)

    # dataset config
    parser.add_argument('--dataset', type=str, default='abus3d')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--fold', type=str, default='0')

    # optimizer config
    parser.add_argument('--lr_scheduler', type=str, default='cos', choices=['poly', 'step', 'cos'])
    parser.add_argument('--lr', default=1e-3, type=float) 
    parser.add_argument('--weight_decay', default=1e-5, type=float)

    # network config
    parser.add_argument('--arch', default='attd2unet', type=str, choices=('attd2unet', 'd2unet'))

    # losses 
    parser.add_argument('--loss', type=str, default='dice', choices=('dice', 'focal_dice', 'boundary', 'focal', 'focalti', 'fdl', 'ours'))
    parser.add_argument('--boundary_method', type=str, default='rump', choices=('rump', 'stable'))
    parser.add_argument('--boundary_alpha', type=float, default=0.001)
    parser.add_argument('--gamma', type=float, default=2)

    parser.add_argument('--beta', type=float, default=0.1)
    parser.add_argument('--use_dismap', action='store_true')

    # save config
    parser.add_argument('--log_dir', default='./log')
    parser.add_argument('--save', default='./work/train/uxnet')

    args = parser.parse_args()
    return args


def main():
    #############
    # init args #
    #############
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu 

    # creat save path
    if os.path.exists(args.save):
        shutil.rmtree(args.save)
    os.makedirs(args.save, exist_ok=True)

    # logger
    logging.basicConfig(filename=args.save+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info('--- init parameters ---')

    # writer
    idx = args.save.rfind('/')
    log_dir = args.log_dir + args.save[idx:]
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    writer = SummaryWriter(log_dir)

    # set title of the current process
    #setproctitle.setproctitle('auto-attention-train')

    # random
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    #####################
    # building  network #
    #####################
    logging.info('--- building network ---')

    if args.arch == 'attd2unet':
        model = AttD2UNet()
    elif args.arch == 'd2unet':
        model = D2UNet()
    else:
        raise(NotImplementedError('model {} not implement'.format(args.arch))) 
    n_params = sum([p.data.nelement() for p in model.parameters()])
    logging.info('--- total parameters = {} ---'.format(n_params))

    model = model.cuda()

    ################
    # prepare data #
    ################
    logging.info('--- loading dataset ---')
    kwargs = {'num_workers': args.num_workers, 
              'pin_memory': True,}
    train_loader, _, val_loader = make_data_loader(args, **kwargs)
    
    #####################
    # optimizer & loss  #
    #####################
    logging.info('--- configing optimizer & losses ---')
    lr = args.lr
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    lr_scheduler = LR_Scheduler(args.lr_scheduler, args.lr, args.epoch, len(train_loader))

    loss_fn = {}
    loss_fn['dice_loss'] = dice_loss
    loss_fn['l1_loss'] = nn.L1Loss()

    loss_fn['focalti'] = FocalTiLoss()
    loss_fn['fdl'] = FocalDiceLoss(gamma=args.gamma)
    loss_fn['focal'] = FocalLoss()
    loss_fn['boundary'] = boundary_loss


    #####################
    #   strat training  #
    #####################
    logging.info('--- start training ---')

    best_pre = 0.
    nTrain = len(train_loader.dataset)
    for epoch in range(args.start_epoch, args.epoch + 1):
        train(args, epoch, model, train_loader, optimizer, loss_fn, writer, lr_scheduler)
        dice = val(args, epoch, model, val_loader, optimizer, loss_fn, writer)
        is_best = False
        if dice > best_pre:
            is_best = True
            best_pre = dice
        save_checkpoint({'epoch': epoch,
                         'state_dict': model.state_dict(),
                         'best_pre': best_pre},
                          is_best, 
                          args.save, 
                          args.arch)
    writer.close()


def train(args, epoch, model, train_loader, optimizer, loss_fn, writer, lr_scheduler):
    model.train()
    nProcessed = 0
    batch_size = args.ngpu * args.batch_size
    nTrain = len(train_loader.dataset)
    loss_list = []

    for batch_idx, sample in enumerate(train_loader):
        # read data
        image, target = sample['image'], sample['target']
        image, target = Variable(image.cuda()), Variable(target.cuda(), requires_grad=False)

        # forward
        seg_pred = model(image)
        seg_pred_soft = F.softmax(seg_pred, dim=1)
        #print(seg_pred.shape)
        #print(target.shape)
        #loss = loss_fn['dice_loss'](seg_pred[:, 1, ...], target==1)
        if args.loss == 'dice':
            loss_dice = loss_fn['dice_loss'](seg_pred_soft[:, 1, ...], target == 1)
            loss = loss_dice
        elif args.loss == 'fdl':
            loss_fdl = loss_fn['fdl'](seg_pred_soft[:, 1, ...], target == 1)
            loss = loss_fdl
        elif args.loss == 'focal':
            loss_focal = loss_fn['focal'](seg_pred, target.long())
            loss = loss_focal
        elif args.loss == 'focalti':
            loss_focal_ti = loss_fn['focalti'](seg_pred_soft[:, 1, ...], target == 1)
            loss = loss_focal_ti
        elif args.loss == 'boundary':
            loss_dice = loss_fn['dice_loss'](seg_pred_soft[:, 1, ...], target == 1)

            dist = sample['dist']
            dist = dist.cuda()
            loss_boundary = loss_fn['boundary'](seg_pred_soft, dist)

            if args.boundary_method == 'rump':
                alpha = 0.005 * epoch
                if alpha >= 0.9:
                    alpha = 0.9
                loss = (1 - alpha) * loss_dice + alpha * loss_boundary
            else:
                loss = loss_dice + args.boundary_alpha * loss_boundary

        # backward
        lr = lr_scheduler(optimizer, batch_idx, epoch, 0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_list.append(loss.item())
        writer.add_scalar('lr', lr, epoch)

        # visualization
        nProcessed += len(image)
        partialEpoch = epoch + batch_idx / len(train_loader)
        print('Train Epoch: {:.2f} [{}/{} ({:.0f}%)]\tLoss: {:.8f}'.format(
            partialEpoch, nProcessed, nTrain, 100. * batch_idx / len(train_loader),
            loss.item()))

    writer.add_scalar('train_loss',float(np.mean(loss_list)), epoch)


def val(args, epoch, model, val_loader, optimizer, loss_fn, writer):
    model.eval()
    dice_list = []

    with torch.no_grad():
        for batch_idx, sample in tqdm.tqdm(enumerate(val_loader)):
            image, target = sample['image'], sample['target']
            image, target = image.cuda(), target.cuda()

            # forward
            seg_pred = model(image)
            seg_pred = F.softmax(seg_pred, dim=1)
            seg_pred = seg_pred.max(1)[1]
            dice = get_dice(seg_pred.cpu().numpy(), target.cpu().numpy())
            dice_list.append(dice)

        writer.add_scalar('val_dice', float(np.mean(dice_list)), epoch)
        return np.mean(dice_list)


if __name__ == '__main__':
    main()
