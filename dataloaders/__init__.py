import torch.utils.data as data
import torchvision.transforms as transforms
import random

def make_data_loader(args, **kwargs):
    if args.dataset == 'isic':
        from dataloaders.isic_dataset import ISICDataset, ToTensor, Normalize 
        root_path = '/data/xuyangcao/code/data/isic/'  

        train_transform = transforms.Compose([
            ToTensor(mode='train'), 
            Normalize(0.5, 0.5, mode='train')
            ])
        val_transform = transforms.Compose([
            ToTensor(mode='val'), 
            Normalize(0.5, 0.5, mode='val')
            ])
        test_transform = transforms.Compose([
            ToTensor(mode='test'), 
            Normalize(0.5, 0.5, mode='test')
            ])

        train_set = ISICDataset(base_dir=root_path, mode='train', transform=train_transform) 
        valid_dataset = ISICDataset(base_dir=root_path, mode='val', transform=val_transform)
        test_dataset = ISICDataset(base_dir=root_path, mode='test', transform=val_transform)

        train_loader = data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = data.DataLoader(test_dataset, batch_size=1, shuffle=False, **kwargs)
        return train_loader, val_loader, test_loader

    elif args.dataset == 'abus3d':
        from dataloaders.abus_dataset_3d import ABUSDataset, RandomFlip, ToTensor
        root_path = '/data/xuyangcao/code/data/roi_3d/abus_shift'  

        train_transform = transforms.Compose([
            RandomFlip(probability=0.2, use_dismap=args.use_dismap),
            #ElasticTransform(probability=0.2),
            #RandomCrop(output_size=cfg.general.crop_size),
            ToTensor(use_dismap=args.use_dismap)
            ])
        val_transform = transforms.Compose([
            RandomFlip(probability=0.2),
            ToTensor()
            ])
        test_transform = transforms.Compose([ToTensor()])

        train_dataset = ABUSDataset(
                base_dir=root_path, 
                split='train', 
                fold=args.fold, 
                transform=train_transform,
                use_dismap=args.use_dismap,
                beta=args.beta
                )
        valid_dataset = ABUSDataset(
                base_dir=root_path, 
                split='val', 
                fold=args.fold, 
                transform=train_transform
                )
        test_dataset = ABUSDataset(
                base_dir=root_path, 
                split='test', 
                fold=args.fold, 
                transform=test_transform
                )
        train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = data.DataLoader(test_dataset, batch_size=1, shuffle=False, **kwargs)
        return train_loader, val_loader, test_loader

    else:
        from dataloaders.abus_dataset import ABUS_2D, ElasticTransform, ToTensor, Normalize
        #root_path = '/home/xuyangcao/code/data/abus_2d/'
        root_path = '/data/xuyangcao/code/data/abus_2d/'
        sample_k = 8856 #100, 300, 885, 1770, 4428, 8856

        train_transform = transforms.Compose([
            ElasticTransform('train'), 
            ToTensor(mode='train'), 
            Normalize(0.5, 0.5, mode='train')
            ])
        val_transform = transforms.Compose([
            ElasticTransform(mode='val'),
            ToTensor(mode='val'), 
            Normalize(0.5, 0.5, mode='val')
            ])
        test_transform = transforms.Compose([
            ToTensor(mode='test'), 
            Normalize(0.5, 0.5, mode='test')
            ])
        train_set = ABUS_2D(base_dir=root_path,
                            mode='train', 
                            data_num_labeled=sample_k,
                            use_unlabeled_data=False,
                            transform=train_transform
                            )
        val_set = ABUS_2D(base_dir=root_path,
                           mode='val', 
                           data_num_labeled=None, 
                           use_unlabeled_data=False, 
                           transform=val_transform
                           )
        test_set = ABUS_2D(base_dir=root_path,
                           mode='test', 
                           data_num_labeled=None, 
                           use_unlabeled_data=False, 
                           transform=val_transform
                           )
        batch_size = args.ngpu * args.batch_size
        train_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=True, **kwargs)
        #val_loader = data.DataLoader(val_set, batch_size=1, shuffle=True, **kwargs)
        val_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=True, **kwargs)
        test_loader = data.DataLoader(test_set, batch_size=1, shuffle=False, **kwargs)

        return train_loader, val_loader, test_loader

    
