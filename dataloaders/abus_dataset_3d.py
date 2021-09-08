import os 
import numpy as np
import random
import SimpleITK as sitk

import torch 
from torch.utils.data import Dataset 
from scipy.ndimage import distance_transform_edt as distance


def target2dist(target, loss='boundary', beta=0.5):
    target[target != 0] = 1

    if loss == 'boundary':
        posmask = target.astype(np.bool)
        negmask = ~posmask
        dist = distance(negmask) * negmask - (distance(posmask) - 1) * posmask
        dist = dist.astype(np.float32)
    else:
        dist = target.copy()
        dist = dist.astype(np.float32)
        dist[dist != 0] = -beta
        dist[dist == 0] = 1 - beta

    return dist


class ABUSDataset(Dataset):
    """ abus dataset """
    def __init__(self, base_dir=None, split='train', fold='0', num=None, use_dismap=False, transform=None, loss='boundary', beta=0.5):
        self._base_dir = base_dir
        self.transform = transform
        self.sample_list = []
        self.split = split
        self.use_dismap = use_dismap 
        self.loss = loss
        self.beta = beta

        if split=='train':
            with open(self._base_dir+'/../abus_train.list.'+fold, 'r') as f:
                self.image_list = f.readlines()
        elif split == 'test':
            with open(self._base_dir+'/../abus_test.list.'+fold, 'r') as f:
                self.image_list = f.readlines()
        elif split == 'val':
            with open(self._base_dir+'/../abus_val.list.'+fold, 'r') as f:
                self.image_list = f.readlines()
        self.image_list = [item.replace('\n','') for item in self.image_list]

        if num is not None:
            self.image_list = self.image_list[:num]
        print("--- total {} samples ---".format(len(self.image_list)))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        image = self.load_img('image/'+image_name, is_normalize=True)
        target = self.load_img('label/'+image_name)
        target[target!=0] = 1

        if self.use_dismap:
            #dis_map = self.load_img('signed_geo_map/'+image_name[:-3]+'nii.gz')
            dist = target2dist(target, self.loss, self.beta)
            sample = {'image': image, 'target': target, 'dist': dist}
        else:
            sample = {'image': image, 'target': target}

        #print('image.shape', image.shape)
        if self.transform:
            sample = self.transform(sample)
        
        sample['file_name'] = image_name
        return sample
    
    def load_img(self, image_name, is_normalize=False):
        filename = os.path.join(self._base_dir, image_name)
        #print(image_name)
        itk_img = sitk.ReadImage(filename)
        image = sitk.GetArrayFromImage(itk_img)
        #image = np.transpose(image, (1,2,0))
        image = image.astype(np.float32)
        #print('image.shape: ', image.shape)

        if is_normalize:
            #print('image.max ', image.max())
            #print('image.min', image.min())
            image = image / image.max() 
            image = (image - 0.5) / 0.5

        return image

class ElasticTransform(object):
    r"""
    Elastic transform using b-spline interplation
    """
    def __init__(self, probability=0.5, use_dismap=False):
        self.probability = probability
        self.use_dismap = use_dismap

    def __call__(self, sample):
        image, target = sample['image'], sample['target']

        prob = round(np.random.uniform(0, 1), 1) 
        if prob < self.probability:
            num_control_points = np.random.randint(20, 32)
            image, target = self._produceRandomlyDeformedImage(image, target, num_control_points)
            #print('--- elastic: {} ---'.format(prob))

        sample['image'], sample['target'] = image, target
        return sample

class RandomFlip(object):
    r""" random flip image along single and multi-axis.

    """
    def __init__(self, probability=0.5, use_dismap=False):
        self.probability = probability
        self.flip_dict = [
                [0],
                [1],
                [2],
                [0, 1],
                [0, 2],
                [1, 2],
                [0, 1, 2]
                ] 
        self.use_dismap = use_dismap

    def __call__(self, sample):
        image, target = sample['image'], sample['target']

        prob = round(np.random.uniform(0, 1), 1) 
        if prob < self.probability:
            idx = random.randint(0, 6)
            image = np.flip(image, axis=self.flip_dict[idx]).copy()
            target = np.flip(target, axis=self.flip_dict[idx]).copy()
            if self.use_dismap:
                dist = sample['dist']
                dist = np.flip(dist, axis=self.flip_dict[idx]).copy()
                sample['dist'] = dist
            #print('--- flip: {} ---'.format(prob))
            sample['image'], sample['target'] = image, target
        return sample


class RandomNoise(object):
    r""" add noise to input image

    """
    def __init__(self, mu=0, sigma=0.1):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, sample):
        image, target = sample['image'], sample['target']
        noise = np.clip(self.sigma * np.random.randn(image.shape[0], image.shape[1], image.shape[2]), -2*self.sigma, 2*self.sigma)
        noise = noise + self.mu
        image = image + noise

        return {'image': image, 'target': target}


class ToTensor(object):
    r"""Convert ndarrays in sample to Tensors.

    """
    def __init__(self, use_dismap=False):
        self.use_dismap = use_dismap

    def __call__(self, sample):
        image, target = sample['image'], sample['target']
        image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2]).astype(np.float32)
        sample['image'] = torch.from_numpy(image)
        #print('target.shape: ', target.shape)
        sample['target'] = torch.from_numpy(target).long()

        if self.use_dismap:
            dist = sample['dist']
            dist = np.expand_dims(dist, 0)
            #print('dis_map.shape: ', dis_map.shape)
            sample['dist'] = torch.from_numpy(dist.astype(np.float32))

        return sample
