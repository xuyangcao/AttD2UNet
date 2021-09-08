#python train.py --save ./work/attd2unet_skiponly --gpu 0
#python train.py --save ./work/d2unet --gpu 2 --arch d2unet 

#python train.py --save ./work/attd2unet_fdl --gpu 3 --arch attd2unet --loss fdl
#python train.py --save ./work/attd2unet_fti --gpu 2 --arch attd2unet --loss focalti

#python train.py --save ./work/attd2unet_boundary --gpu 1 --arch attd2unet --loss boundary --use_dismap
python train.py --save ./work/attd2unet_focal --gpu 1 --arch attd2unet --loss focal
