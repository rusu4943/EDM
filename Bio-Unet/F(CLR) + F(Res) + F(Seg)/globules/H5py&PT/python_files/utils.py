import torch
import math
import numpy as np
from scipy import stats
from torchvision import transforms

class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class IoUMeter(object):

    def __init__( self, threshold=0.3 ):
        self.smooth = 1e-15
        self.threshold = threshold
        self.intersection = torch.zeros(5)
        self.union = torch.zeros(5)
        # per image stats for t-test
        self.per_image_int = []
        self.per_image_union = []
        self.per_image_iou = []

    def update( self, prob_mask, gt_mask ):
        pred_mask = ( prob_mask > self.threshold ).type( gt_mask.dtype )
        # shape: batch_size x num_attrs
        self.intersection += (pred_mask * gt_mask).sum(dim=[0,2,3])
        self.union += pred_mask.sum(dim=[0,2,3]) + gt_mask.sum(dim=[0,2,3])
        # per image stats
        # import pdb; pdb.set_trace()
        self.per_image_int += (pred_mask * gt_mask).sum(dim=[1,2,3]).tolist()
        self.per_image_union += \
                ( pred_mask.sum(dim=[1,2,3]) + gt_mask.sum(dim=[1,2,3]) ).tolist()

    def mean_iou( self ):
        # save per image iou stats for t-test
        self.update_per_image_iou()
        # ISIC has a really stupid way of calculating iou:
        # https://challenge2018.isic-archive.com/task2/
        miou = self.intersection.sum() / \
                ( self.union.sum() - self.intersection.sum() + self.smooth )
        return miou

    def iou_per_attr( self ):
        iou_pa = self.intersection / ( self.union - 
                self.intersection + self.smooth )
        return iou_pa

    def update_per_image_iou( self ):
        # import pdb; pdb.set_trace()
        eps = 1e-6
        per_image_int = np.array( self.per_image_int )
        per_image_union = np.array( self.per_image_union )
        self.per_image_iou = ( per_image_int + eps ) / ( per_image_union + eps )
        return self.per_image_iou

class CorrMeter(object):

    def __init__( self, threshold=0.3 ):
        self.threshold = threshold
        self.resize = transforms.Resize(7)
        self.per_image_corr = []

    def update( self, prob_mask, gt_mask ):
        # import pdb; pdb.set_trace()
        prob_mask, gt_mask = self.resize(prob_mask), self.resize(gt_mask)
        pred_mask = ( prob_mask > self.threshold ).type( gt_mask.dtype )
        for i in range( len( gt_mask ) ):
            corr, _ = stats.pearsonr(pred_mask[i].detach().flatten().cpu(),
                    gt_mask[i].detach().flatten().cpu())
            corr = 1 if np.isnan( corr ) else corr
            self.per_image_corr.append( corr )

    def mean_corr( self ):
        # import pdb; pdb.set_trace()
        m_corr = np.array( self.per_image_corr ).mean()
        return m_corr

def accuracy( pred, labels ):
    assert pred.shape == labels.shape
    labels = labels
    pred = pred > 0.5
    correct = ( pred == labels ).sum().item()
    total = labels.nelement()
    acc = correct / total
    return acc

def Dice( preds, masks ):
    smooth = 1.
    iflat = preds.contiguous().view(-1)
    tflat = masks.contiguous().view(-1)
    intersection = (iflat * tflat).sum()
    A_sum = torch.sum(iflat * iflat)
    B_sum = torch.sum(tflat * tflat)
    return (2. * intersection + smooth) / (A_sum + B_sum + smooth) 

def deprocess_image(img):
    """
    see https://github.com/jacobgil/keras-grad-cam/blob/master/grad-cam.py#L65
    """
    img = img - np.mean(img)
    img = img / (np.std(img) + 1e-5)
    img = img * 0.1
    img = img + 0.5
    img = np.clip(img, 0, 1)
    return np.uint8(img * 255)

def ttest_iou( iou_meters_per_threshold ):
    """
    :param iou_meters_per_threshold: dict of dict. First level
    key is threshold. Second level key is model_id
    :returns: ttest scores by threshold
    """
    # import pdb; pdb.set_trace()
    ttest_scores = {}
    avg_model_id = 'model_avg'
    for thresh in iou_meters_per_threshold:
        # ttest_scores[thresh] = {}
        avg_iou_meter = iou_meters_per_threshold[thresh][avg_model_id]
        avg_per_image_iou = avg_iou_meter.per_image_iou
        ens_per_image_iou = []
        for model_id, iou_meter in iou_meters_per_threshold[thresh].items():
            if model_id == avg_model_id: continue
            ens_per_image_iou.append( avg_per_image_iou - iou_meter.per_image_iou )
            # res = stats.ttest_ind( avg_per_image_iou, m_per_image_iou )
            # ttest_scores[thresh][model_id] = res.pvalue
        ens_per_image_iou = np.concatenate( ens_per_image_iou )
        # res = stats.ttest_ind( avg_per_image_iou, ens_per_image_iou )
        res = stats.ttest_1samp( ens_per_image_iou, 0, alternative='greater' )
        mean, std = ens_per_image_iou.mean(), ens_per_image_iou.std()
        ttest_scores[ thresh ] = ( res.pvalue, mean, std )
    return ttest_scores

def ttest_corr( corr_meters_per_threshold ):
    """
    :param corr_meters_per_threshold: dict of dict. First level
    key is threshold. Second level key is model_id
    :returns: ttest scores by threshold
    """
    # import pdb; pdb.set_trace()
    ttest_scores = {}
    avg_model_id = 'model_avg'
    for thresh in corr_meters_per_threshold:
        # ttest_scores[thresh] = {}
        avg_corr_meter = corr_meters_per_threshold[thresh][avg_model_id]
        avg_per_image_corr = avg_corr_meter.per_image_corr
        ens_per_image_corr = []
        for model_id, corr_meter in corr_meters_per_threshold[thresh].items():
            if model_id == avg_model_id: continue
            ens_per_image_corr.append( np.array(avg_per_image_corr) 
                    - np.array(corr_meter.per_image_corr) )
            # res = stats.ttest_ind( avg_per_image_corr, m_per_image_corr )
            # ttest_scores[thresh][model_id] = res.pvalue
        ens_per_image_corr = np.concatenate( ens_per_image_corr )
        # res = stats.ttest_ind( avg_per_image_corr, ens_per_image_corr )
        res = stats.ttest_1samp( ens_per_image_corr, 0, alternative='greater' )
        mean, std = ens_per_image_corr.mean(), ens_per_image_corr.std()
        ttest_scores[ thresh ] = ( res.pvalue, mean, std )
    return ttest_scores
