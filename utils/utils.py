import os
import cv2
import sys
import time
import torch
import shutil
import logging
import importlib
from medpy import metric
import numpy as np
from skimage.transform import resize



def one_hot(gt):
    gt = gt.long()
    gt_one_hot = torch.eye(2)[gt.squeeze(1)]
    gt_one_hot = gt_one_hot.permute(0, 3, 1, 2).float()
    if gt.cuda:
        gt_one_hot = gt_one_hot.cuda()

    return gt_one_hot

def get_labeled_data(data, labels):
        cond = labels[:, 0, 0, 0] >= 0 # first element of all samples in a batch 
        nnz = torch.nonzero(cond) # get how many labeled samples in a batch 
        nbsup = len(nnz)
        #print('labeled samples number:', nbsup)
        if nbsup > 0:
            masked_data = data[cond]
            masked_labels = labels[cond]
            return masked_data, masked_labels
        else:
            return None, None
    
def get_unlabeled_data(data, labels):
        cond = labels[:, 0, 0, 0] < 0
        nnz = torch.nonzero(cond)
        nbsup = len(nnz)
        if nbsup > 0:
            masked_data = data[cond]
            masked_labels = labels[cond]
            return masked_data, masked_labels
        else:
            return None, None


def confusion(y_pred, y_true):
    '''
    get precision and recall
    '''
    y_pred = y_pred.float().view(-1) 
    y_true = y_true.float().view(-1)
    smooth = 1. 
    y_pred_pos = y_pred
    y_pred_neg = 1 - y_pred_pos
    y_pos = y_true
    y_neg = 1 - y_true

    tp = torch.dot(y_pos, y_pred_pos)
    fp = torch.dot(y_neg, y_pred_pos)
    fn = torch.dot(y_pos, y_pred_neg)

    prec = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)

    return prec, recall

def draw_results(img, label, pred):
    _, contours, _ = cv2.findContours(label, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(img, contours, -1, (0, 255, 0), 2)

    pred = resize(pred, label.shape).astype(label.dtype)
    _, contours, _ = cv2.findContours(pred, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img, contours, -1, (255, 0, 0), 2)

    return img

def save_checkpoint(state, is_best, path, arch, filename='checkpoint.pth.tar'):
    filename = 'checkpoint_' + str(state['epoch']) + '.pth.tar'
    #filename = 'checkpoint.pth.tar'
    prefix_save = os.path.join(path, arch)
    checkpoint_name = prefix_save + '_' + filename
    torch.save(state, checkpoint_name)

    if is_best:
        shutil.copyfile(checkpoint_name, prefix_save + '_model_best.pth.tar')


def gaussian_noise(x, batchsize, input_shape=(3, 224, 224), std=0.03):
    noise = torch.zeros(x.shape)
    noise.data.normal_(0, std)
    noise = noise.to(x.device)
    return x + noise

def datestr():
    now = time.gmtime()
    return '{}{:02}{:02}_{:02}{:02}'.format(now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min)

def get_metrics(pred, gt, voxelspacing=(0.5, 0.5, 0.5)):
    r""" 
    Get statistic metrics of segmentation

    These metrics include: Dice, Jaccard, Hausdorff Distance, 95% Hausdorff Distance, 
    and Average surface distance(ASD) metric.

    If the prediction result is 0s, we set hd, hd95, asd 10.0 to avoid errors.

    Parameters:
    -----------
    pred: 3D numpy ndarray
        binary prediction results 

    gt: 3D numpy ndarray
        binary ground truth

    voxelspacing: tuple of 3 floats. default: (0.5, 0.5, 0.5)
        voxel space of 3D image

    Returns:
    --------
    metrics: dict of 5 metrics 
        dict{dsc, jc, hd, hd95, asd}
    """

    dsc = metric.binary.dc(pred, gt)
    jc = metric.binary.jc(pred, gt)
    vs = np.sum(gt) * voxelspacing[0] * voxelspacing[1] * voxelspacing[2]
    precision = metric.binary.precision(pred, gt)
    recall = metric.binary.recall(pred, gt)

    if np.sum(pred) == 0:
        print('=> prediction is 0s! ')
        hd = 32.
        hd95 = 32.
        asd = 10.
    else:
        hd = metric.binary.hd(pred, gt, voxelspacing=voxelspacing)
        hd95 = metric.binary.hd95(pred, gt, voxelspacing=voxelspacing)
        asd = metric.binary.asd(pred, gt, voxelspacing=voxelspacing)

    metrics = {'dsc': dsc, 'jc': jc, 'hd': hd, 'hd95': hd95, 'asd': asd, 
                'precision':precision, 'recall':recall, 'vs': vs} 
    return metrics 

def get_metrics_2d(pred, gt, voxelspacing=(0.21, 0.21)):
    r""" 
    Get statistic metrics of segmentation

    These metrics include: Dice, Jaccard, Hausdorff Distance, 95% Hausdorff Distance, 
    , Pixel Wise Accuracy, Precision and Recall.

    If the prediction result is 0s, we set hd, hd95, 10.0 to avoid errors.

    Parameters:
    -----------

        pred: 2D numpy ndarray
            binary prediction results 

        gt: 2D numpy ndarray
            binary ground truth

        voxelspacing: tuple of 2 floats. default: (0.21, 0.21)
            voxel space of 2D image

    Returns:
    --------

        metrics: dict of 7 metrics 
            dict{dsc, jc, hd, hd95, precision, recall, acc}

    """

    dsc = metric.binary.dc(pred, gt)
    jc = metric.binary.jc(pred, gt)
    precision = metric.binary.precision(pred, gt)
    recall = metric.binary.recall(pred, gt)

    acc = (pred == gt).sum() / len(gt.flatten())

    if np.sum(pred) == 0:
        #print('=> prediction is 0s! ')
        hd = 10 
        hd95 = 10
    else:
        hd = metric.binary.hd(pred, gt, voxelspacing=voxelspacing)
        hd95 = metric.binary.hd95(pred, gt, voxelspacing=voxelspacing)

    metrics = {'dsc': dsc, 'jc': jc, 'hd': hd, 'hd95': hd95, 
                'precision':precision, 'recall':recall, 'acc':acc} 
    return metrics 

def get_dice(pred, gt):
    dice = metric.binary.dc(pred, gt)

    return dice


def load_config(file_name):
    r"""
    load configuration file as a python module

    Args:
        file_name: configuration file name

    Returns:
        a loaded network module            
    """

    dir_name = os.path.dirname(file_name)
    base_name = os.path.basename(file_name)
    module_name, _ = os.path.splitext(base_name)
    #print('dir_name: ', dir_name)
    #print('base_name: ', base_name)
    #print('module_name: ', module_name)

    os.sys.path.append(dir_name)
    config = importlib.import_module(module_name)
    if module_name in sys.modules:
        importlib.reload(config)
    del os.sys.path[-1]

    return config.cfg


if __name__ == "__main__":
    cfg = load_config('../config/config.py')
    print(cfg)
