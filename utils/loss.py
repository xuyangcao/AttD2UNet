import torch
from torch.autograd import Function
import torch.nn.functional as F 
import torch.nn as nn 
from itertools import repeat
import numpy as np
from torch.autograd import Variable
from skimage.measure import label, regionprops
from scipy.ndimage import distance_transform_edt as distance

def boundary_loss(outputs_soft, gt_sdf):
    """
    compute boundary loss for binary segmentation
    input: outputs_soft: sigmoid results,  shape=(b,2,x,y,z)
           gt_sdf: sdf of ground truth (can be original or normalized sdf); shape=(b,2,x,y,z)
    output: boundary_loss; sclar
    """
    pc = outputs_soft[:, [1,], ...]
    dc = gt_sdf[:, [0,], ...] # the shape of distmap here is (b, 1, x, y, x), so we select index = 0
    multipled = torch.einsum('bkxyz, bkxyz->bkxyz', pc, dc)
    bd_loss = multipled.mean()

    return bd_loss

class FocalDiceLoss(nn.Module):
    def __init__(self, gamma=2.):
        super().__init__()
        self.gamma = gamma

    def forward(self, score, target):
        target = target.float()
        smooth = 1e-6
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        dice = (2 * intersect + smooth) / (z_sum + y_sum + smooth)

        loss = (1 - dice**(1. / self.gamma))
        return loss

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        target = target.long()

        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        #print('target.shape: ', target.shape)
        target = target.view(-1,1)

        #print(target.dtype)
        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()

class FocalTiLoss(nn.Module):
    # Tversky index loss
    def __init__(self, alpha=0.7, beta=0.4, gamma=0.75):
        super().__init__()
        self.alpha = alpha
        self.beta = beta 
        self.gamma = gamma
        self.eps = 1e-6

    def forward(self, output, target):
        output = output.float()
        target = target.float()

        pi = output.contiguous().view(-1)
        gi = target.contiguous().view(-1)
        p_ = 1 - pi
        g_ = 1 - gi
        
        intersection = torch.dot(pi, gi)
        inter_alpha = torch.dot(p_, gi)
        inter_beta = torch.dot(g_, pi)
        
        ti = (intersection + self.eps) / (intersection + self.alpha*inter_alpha + self.beta*inter_beta + self.eps)
        loss = torch.mean(torch.pow(1-ti, self.gamma))
        return loss

def dice_loss(score, target):
    #print(score.dtype)
    #print(target.dtype)
    target = target.float()
    score = score.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target * target)
    z_sum = torch.sum(score * score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss

def loss_calc(pred, label):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    label = Variable(label.long()).cuda()
    criterion = CrossEntropy2d().cuda()

    return criterion(pred, label)

class CrossEntropy2d(nn.Module):

    def __init__(self, size_average=True, ignore_label=255):
        super(CrossEntropy2d, self).__init__()
        self.size_average = size_average
        self.ignore_label = ignore_label

    def forward(self, predict, target, weight=None):
        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(1), "{0} vs {1} ".format(predict.size(2), target.size(1))
        assert predict.size(3) == target.size(2), "{0} vs {1} ".format(predict.size(3), target.size(3))
        n, c, h, w = predict.size()
        #print('target.max(): ', target.max())
        #print('target.min(): ', target.min())
        target_mask = (target >= 0) * (target != self.ignore_label)
        target = target[target_mask]
        if not target.data.dim():
            return Variable(torch.zeros(1))
        predict = predict.transpose(1, 2).transpose(2, 3).contiguous()
        predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
        loss = F.cross_entropy(predict, target, weight=weight, size_average=self.size_average)
        return loss

class BCEWithLogitsLoss2d(nn.Module):

    def __init__(self, size_average=True, ignore_label=255):
        super(BCEWithLogitsLoss2d, self).__init__()
        self.size_average = size_average
        self.ignore_label = ignore_label

    def forward(self, predict, target, weight=None):
        """
            Args:
                predict:(n, 1, h, w)
                target:(n, 1, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 4
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(2), "{0} vs {1} ".format(predict.size(2), target.size(2))
        assert predict.size(3) == target.size(3), "{0} vs {1} ".format(predict.size(3), target.size(3))
        n, c, h, w = predict.size()
        #print('target.max(): ', target.max())
        #print('target.min(): ', target.min())
        target_mask = (target >= 0) * (target != self.ignore_label)
        target = target[target_mask]
        if not target.data.dim():
            return Variable(torch.zeros(1))
        predict = predict[target_mask]
        loss = F.binary_cross_entropy_with_logits(predict, target, weight=weight, reduction='mean')
        return loss

class MaskDiceLoss(nn.Module):
    def __init__(self):
        super(MaskDiceLoss, self).__init__()

    def dice_loss(self, gt, pre, eps=1e-6):
        num_classes = pre.shape[1]
        # get one hot ground gruth
        gt = gt.long()
        gt_one_hot = torch.eye(num_classes)[gt.squeeze(1)]
        gt_one_hot = gt_one_hot.permute(0, 3, 1, 2).float()
        # dice loss 
        pre = pre.float()
        #print('pre.shape: ', pre.shape)
        #print('gt.shape: ', gt_one_hot.shape)
        if gt.cuda:
            gt_one_hot = gt_one_hot.cuda()
        dims = (0,) + tuple(range(2, gt.ndimension()))
        intersection = torch.sum(pre * gt_one_hot, dims)
        cardinality = torch.sum(pre + gt_one_hot, dims)
        dice = (2. * intersection / (cardinality + eps)).mean()

        return 1 - dice

    def ce_loss(self, gt, pre):
        pre = pre.permute(0,2,3,1).contiguous()
        pre = pre.view(pre.numel() // 2, 2)
        gt = gt.view(gt.numel())
        loss = F.cross_entropy(pre, gt.long())

        return loss

    def forward(self, out, labels):
        labels = labels.float()
        out = out.float()

        cond = labels[:, 0, 0, 0] >= 0 # first element of all samples in a batch 
        nnz = torch.nonzero(cond) # get how many labeled samples in a batch 
        nbsup = len(nnz)
        #print('labeled samples number:', nbsup)
        if nbsup > 0:
            masked_outputs = torch.index_select(out, 0, nnz.view(nbsup)) #select all supervised labels along 0 dimention 
            masked_labels = labels[cond]

            dice_loss = self.dice_loss(masked_labels, masked_outputs)
            #ce_loss = self.ce_loss(masked_labels, masked_outputs)

            loss = dice_loss
            return loss, nbsup
        return Variable(torch.FloatTensor([0.]).cuda(), requires_grad=False), 0

class MaskMSELoss(nn.Module):
    def __init__(self, args):
        super(MaskMSELoss, self).__init__()
        self.args = args

    def forward(self, out, zcomp, uncer, th=0.15):
        # transverse to float 
        out = out.float() # current prediction
        zcomp = zcomp.float() # the psudo label 
        uncer = uncer.float() #current prediction uncertainty
        if self.args.is_uncertain:
            mask = uncer > th
            mask = mask.float()
            mse = torch.sum(mask*(out - zcomp)**2) / torch.sum(mask) 
        else:
            mse = torch.sum((out - zcomp)**2) / out.numel()

        return mse

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pre, gt, eps=1e-6):
        num_classes = pre.shape[1]
        # get one hot ground gruth
        gt = gt.long()
        gt_one_hot = torch.eye(num_classes)[gt.squeeze(1)]
        gt_one_hot = gt_one_hot.permute(0, 3, 1, 2).float()
        # dice loss 
        pre = pre.float()
        #print('pre.shape: ', pre.shape)
        #print('gt.shape: ', gt_one_hot.shape)
        if gt.cuda:
            gt_one_hot = gt_one_hot.cuda()
        dims = (0,) + tuple(range(2, gt.ndimension()))
        intersection = torch.sum(pre * gt_one_hot, dims)
        cardinality = torch.sum(pre + gt_one_hot, dims)
        dice = (2. * intersection / (cardinality + eps)).mean()

        return 1 - dice
    
    @staticmethod
    def dice_coeficient(output, target):
        output = output.float()
        target = target.float()
        
        output = output
        smooth = 1e-20
        iflat = output.view(-1)
        tflat = target.view(-1)
        #print(iflat.shape)
        
        intersection = torch.dot(iflat, tflat)
        dice = (2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth)

        return dice 
