import os 
import tqdm
import shutil 
import argparse
import setproctitle
import pandas as pd
import numpy as np 
from skimage import measure
from skimage.io import imsave
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from ast import literal_eval
import SimpleITK as sitk

import torch 
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from utils.utils import get_metrics, draw_results
from dataloaders import make_data_loader
from models.attention_d2unet import AttD2UNet
from models.denseatt import DenseAtt

def get_args():
    parser = argparse.ArgumentParser()
    # general config
    parser.add_argument('--gpu', default='1', type=str)
    parser.add_argument('--ngpu', type=int, default=1)

    # dataset config
    parser.add_argument('--dataset', type=str, default='abus3d')
    parser.add_argument('--fold', type=str, default='0')
    parser.add_argument('--batch_size', type=int, default=1)

    # network config
    parser.add_argument('--arch', default='attention_d2unet', type=str, choices=('dense161', 'dense121', 'dense201', 'unet', 'resunet', 'sdnet', 'auto_attention'))
    parser.add_argument('--in_channels', default=1, type=int)
    parser.add_argument('--num_classes', default=2, type=int)
    parser.add_argument('--num_init_features', default=64, type=int)
    parser.add_argument('--growth_rate', default=48, type=int)
    parser.add_argument('--bn_size', default=4, type=int)
    parser.add_argument('--block_config', default='[6, 12, 24, 16]', type=str)

    # resume and save
    parser.add_argument('--resume', type=str)
    parser.add_argument('--save', default=None, type=str) # save to resume path if None 
    parser.add_argument('--save_image', action='store_true')

    args = parser.parse_args()
    return args

            
def main():
    # --- init args ---
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu 

    # --- building network ---
    if args.arch == 'attention_d2unet':
        model = AttD2UNet()
    elif args.arch == 'dense_att':
        model = DenseAtt()
    else:
        raise(NotImplementedError('model {} not implement'.format(args.arch))) 
    model = model.cuda()

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_pre = checkpoint['best_pre']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))

            # --- saving path ---
            if 'best' in args.resume:
                file_name = 'model_best_' + str(checkpoint['epoch'])
            elif 'check' in args.resume:
                file_name = 'checkpoint_{}_result'.format(checkpoint['epoch'])

            if args.save is not None:
                save_path = os.path.join(args.save, file_name) 
                csv_path = os.save
            else:
                save_path = os.path.join(os.path.dirname(args.resume), file_name)
                csv_path = os.path.dirname(args.resume)
            setproctitle.setproctitle('...')

            if args.save_image:
                # image path
                args.save_image_path = save_path + '/image' 
                if os.path.exists(args.save_image_path):
                    shutil.rmtree(args.save_image_path)
                os.makedirs(args.save_image_path, exist_ok=True)
                # label path
                args.save_pred_path = save_path + '/pred' 
                if os.path.exists(args.save_pred_path):
                    shutil.rmtree(args.save_pred_path)
                os.makedirs(args.save_pred_path, exist_ok=True)
                args.save_path = save_path
                print('=> saving images in :', save_path)
            else:
                print('we don\'t save any images!')

            # xlsx path
            csv_file_name = file_name + '.xlsx'
            args.csv_file_name = os.path.join(csv_path, csv_file_name) 
            print('=> saving csv in :', args.csv_file_name)

        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        raise(RuntimeError('resume is None!'))


    # --- preparing dataset
    kwargs = {'num_workers': 1, 'pin_memory': True}
    _, _, test_loader = make_data_loader(args, **kwargs)


    # --- testing ---
    test(args, test_loader, model)

def test(args, loader, model):
    model.eval()
    
    dsc_list = []
    jc_list = []
    hd_list = []
    hd95_list = []
    asd_list = []
    precision_list = []
    recall_list = []
    vs_list = []
    filename_list = []

    with torch.no_grad():
        for sample in tqdm.tqdm(loader):
            image, label, file_name = sample['image'], sample['target'], sample['file_name']

            image = image.cuda()
            if args.arch == 'sdnet':
                _, _, _, _, _, pred, _, _, _, _ = model(image, label, 'val')
            else:
                pred = model(image)
                pred = F.softmax(pred, dim=1)
            pred = pred.max(1)[1]

            image = image[0][0].cpu().numpy()
            image = (image + 0.5) * 0.5
            image = image.astype(np.float)

            label = label[0].cpu().numpy()
            label = label.astype(np.float)
            pred = pred[0].cpu().numpy()
            pred = pred.astype(np.float)

            # get metrics
            metrics = get_metrics(pred, label, voxelspacing=(0.5, 0.5, 0.5)) 
            dsc_list.append(metrics['dsc'])
            jc_list.append(metrics['jc'])
            hd_list.append(metrics['hd'])
            hd95_list.append(metrics['hd95'])
            asd_list.append(metrics['asd'])
            precision_list.append(metrics['precision'])
            recall_list.append(metrics['recall'])
            vs_list.append(metrics['vs'])
            filename_list.append(file_name)


            if args.save_image:
                save_name = os.path.join(args.save_path, file_name[0][:-4])
                if not os.path.exists(save_name):
                    os.makedirs(save_name)

                img = sitk.GetImageFromArray(image)
                sitk.WriteImage(img, save_name + '/' + "img.nii.gz")

                img = sitk.GetImageFromArray(label)
                sitk.WriteImage(img, save_name + '/' + "gt.nii.gz")

                img = sitk.GetImageFromArray(pred)
                sitk.WriteImage(img, save_name + '/' + "pred.nii.gz")

        df = pd.DataFrame()
        df['filename'] = filename_list 
        df['dsc'] = np.array(dsc_list)
        df['jc'] = np.array(jc_list) 
        df['hd'] = np.array(hd_list) 
        df['precision'] = np.array(precision_list)
        df['recall'] = np.array(recall_list)
        print(df.describe())
        df['hd95'] = np.array(hd95_list) 
        df['asd'] = np.array(asd_list) 
        df['volume(mm^3)'] = np.array(vs_list)
        df.to_excel(args.csv_file_name)

if __name__ == '__main__':
    main()
