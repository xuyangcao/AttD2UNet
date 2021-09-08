import os
import numpy as np
from skimage.io import imread 

import torch
import torch.utils.data as data 


class ISICDataset(data.Dataset):
    def __init__(self, base_dir=None, mode='train', transform=None):
        self._base_dir = base_dir
        self._transform = transform
        self._mode = mode

        # read list of train or test images
        if mode == 'train':
            with open(self._base_dir + 'lists/train.list', 'r') as f:
                self.image_list = f.readlines()
            self.image_list = [item.replace('\n', '') for item in self.image_list]
        elif mode == 'test':
            with open(self._base_dir + 'lists/test.list', 'r') as f:
                self.image_list = f.readlines()
            self.image_list = [item.replace('\n', '') for item in self.image_list]
        elif mode == 'val':
            with open(self._base_dir + 'lists/val.list', 'r') as f:
                self.image_list = f.readlines()
            self.image_list = [item.replace('\n', '') for item in self.image_list]
        else:
            raise(RuntimeError('mode {} doesn\'t exists!'.format(mode)))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        file_name = self.image_list[idx]
        if self._mode == 'train' or self._mode == 'val':
            # get image
            full_image_name = os.path.join(self._base_dir + '/image/', file_name)
            image = self.load_image(full_image_name, is_normalize=False)

            # get target
            full_target_name = os.path.join(self._base_dir + '/label/', file_name)
            target = self.load_image(full_target_name, is_normalize=False)
            target = target[:, :, 0]
            if target.max() != 1: # transform from 255 to 1
                target[target != 0] = 1.

            # transform 
            sample = {'image': image, 'target': target}
            if self._transform:
                sample = self._transform(sample)

        elif self._mode == 'test':
            # get image
            full_image_name = os.path.join(self._base_dir + '/image/', file_name)
            image = self.load_image(full_image_name, is_normalize=False)

            # get target
            full_target_name = os.path.join(self._base_dir + '/label/', file_name)
            target = self.load_image(full_target_name, is_normalize=False)
            target = target[:, :, 0]
            if target.max() != 1: # transform from 255 to 1
                target[target != 0] = 1.

            # transform 
            sample = {'image':image, 'target':target}
            if self._transform:
                sample = self._transform(sample)

            # return filename to save result in test set
            sample['file_name'] = file_name 
        else:
            raise(RuntimeError('mode {} is invalid in ISICDataset!'.format(self._mode)))

        return sample

    def load_image(self, file_name, is_normalize=False):
        img = imread(file_name) 

        # we don't normalize image when loading them, because Augmentor will raise error
        # if nornalize, normalize origin image to mean=0,std=1.
        if is_normalize:
            img = img.astype(np.float32)
            mean = np.mean(img)
            std = np.std(img)
            img = (img - mean) / std

        return img 

class ToTensor(object):
    def __init__(self, mode='train'):
        self.mode = mode
    
    def __call__(self, sample):
        image, target = sample['image'], sample['target']
        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image)
        target = np.expand_dims(target, 0)
        target = torch.from_numpy(target.astype(np.float32))

        # transverse tensor to 0~1 
        if isinstance(image, torch.ByteTensor): 
            image = image.float().div(255)
        sample['image'], sample['target'] = image, target 

        return sample

class Normalize(object):
    def __init__(self, mean, std, mode='train'):
        self.mode = mode 
        self.mean = mean
        self.std = std
    
    def __call__(self, sample):
        image = sample['image']
        image = (image - self.mean) / self.std

        sample['image'] = image
        return sample
