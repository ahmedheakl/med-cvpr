import numpy as np
import torch
import torch.nn.functional as F
import argparse
import torch.nn as nn
from scipy.ndimage import distance_transform_edt
from skimage import morphology

def boxcount(Z, k):
    """
    returns a count of squares of size kxk in which there are both colours (black and white), ie. the sum of numbers
    in those squares is not 0 or k^2
    Z: np.array, matrix to be checked, needs to be 2D
    k: int, size of a square
    """
    S = np.add.reduceat(
        np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
        np.arange(0, Z.shape[1], k), axis=1)  # jumps by powers of 2 squares

    # We count non-empty (0) and non-full boxes (k*k)
    return len(np.where((S > 0) & (S < k * k))[0])


def fractal_dimension(Z, threshold=0.5):
    """
    calculate fractal dimension of an object in an array defined to be above certain threshold as a count of squares
    with both black and white pixels for a sequence of square sizes. The dimension is the a coefficient to a poly fit
    to log(count) vs log(size) as defined in the sources.
    :param Z: np.array, must be 2D
    :param threshold: float, a thr to distinguish background from foreground and pick up the shape, originally from
    (0, 1) for a scaled arr but can be any number, generates boolean array
    :return: coefficients to the poly fit, fractal dimension of a shape in the given arr
    """
    # Only for 2d image
    assert (len(Z.shape) == 2)

    # Transform Z into a binary array
    Z = (Z < threshold)

    # Minimal dimension of image
    p = min(Z.shape)

    # Greatest power of 2 less than or equal to p
    n = 2 ** np.floor(np.log(p) / np.log(2))

    # Extract the exponent
    n = int(np.log(n) / np.log(2))

    # Build successive box sizes (from 2**n down to 2**1)
    sizes = 2 ** np.arange(n, 1, -1)

    # Actual box counting with decreasing size
    counts = []
    for size in sizes:
        counts.append(boxcount(Z, size))

    # Fit the successive log(sizes) with log (counts)
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    return -coeffs[0]


def compute_hd95(pred, target, pixel_spacing=None):
    """
    Compute the 95th percentile Hausdorff Distance between binary segmentation masks.
    
    Args:
        pred (torch.Tensor): Predicted binary segmentation mask (B, H, W)
        target (torch.Tensor): Ground truth binary segmentation mask (B, H, W)
        pixel_spacing (tuple, optional): Pixel spacing in (y, x) format. Defaults to (1.0, 1.0)
    
    Returns:
        torch.Tensor: 95th percentile Hausdorff Distance
    """
    if pixel_spacing is None:
        pixel_spacing = (1.0, 1.0)

    def compute_surface_distances(mask1, mask2, spacing):
        """Compute surface distances between binary masks."""
        mask1 = mask1.cpu().numpy()
        mask2 = mask2.cpu().numpy()
        
        # Convert to boolean arrays
        mask1 = mask1 > 0.5
        mask2 = mask2 > 0.5
        
        # Distance transforms
        dist1 = distance_transform_edt(~mask1, sampling=spacing)
        dist2 = distance_transform_edt(~mask2, sampling=spacing)
        
        # Get surface points
        surface1 = np.logical_xor(mask1, morphology.binary_erosion(mask1))
        surface2 = np.logical_xor(mask2, morphology.binary_erosion(mask2))
        
        # Get distances from surface points
        distances1 = dist2[surface1]
        distances2 = dist1[surface2]
        
        return distances1, distances2

    def compute_hd95_single(pred_mask, target_mask, spacing):
        """Compute HD95 for a single pair of masks."""
        distances1, distances2 = compute_surface_distances(pred_mask, target_mask, spacing)
        
        if len(distances1) == 0 and len(distances2) == 0:
            return 0.0  # Both masks are empty
        elif len(distances1) == 0 or len(distances2) == 0:
            return np.inf  # One mask is empty
        
        # Compute 95th percentile of distances
        dist1_95 = np.percentile(distances1, 95)
        dist2_95 = np.percentile(distances2, 95)
        
        return max(dist1_95, dist2_95)

    # Handle batch dimension
    if len(pred.shape) == 4:  # (B, C, H, W)
        pred = pred.squeeze(1)
    if len(target.shape) == 4:
        target = target.squeeze(1)
    
    batch_size = pred.shape[0]
    hd95_values = []
    
    for i in range(batch_size):
        hd95 = compute_hd95_single(pred[i], target[i], pixel_spacing)
        hd95_values.append(hd95)
    
    return torch.tensor(np.mean(hd95_values)).to(pred.device)

def dice_coef(y_true, y_pred, smooth=1):
    # print(y_pred.shape, y_true.shape)
    intersection = torch.sum(y_true * y_pred,axis=(-1,-2))
    union = torch.sum(y_true, axis=(-1,-2)) + torch.sum(y_pred, axis=(-1,-2))
    dice = ((2. * intersection + smooth)/(union + smooth)).mean()
    # print(dice)
    return dice

def iou_coef(y_true, y_pred, smooth=1):
    intersection = torch.sum(torch.abs(y_true * y_pred),axis=(-1,-2))
    union = torch.sum(y_true,axis=(-1,-2))+torch.sum(y_pred,axis=(-1,-2))-intersection
    iou = ((intersection + smooth) / (union + smooth)).mean()
    return iou

def running_stats(y_true, y_pred, smooth = 1):
    intersection = torch.sum(y_true * y_pred,axis=(-1,-2))
    union = torch.sum(y_true, axis=(-1,-2)) + torch.sum(y_pred, axis=(-1,-2))
    return intersection, union

def dice_collated(running_intersection, running_union, smooth =1):
    if len(running_intersection.size())>=2:
        dice = (torch.mean((2. * running_intersection + smooth)/(running_union + smooth),dim=1)).sum()
    else:
        dice = ((2. * running_intersection + smooth)/(running_union + smooth)).sum()
    return dice

def dice_batchwise(running_intersection, running_union, smooth =1):
    dice = ((2. * running_intersection + smooth)/(running_union + smooth))
    return dice

def dice_loss(y_pred, y_true):
    numerator = (2 * torch.sum(y_true * y_pred))
    denominator = torch.sum(y_true + y_pred)

    return 1 - ((numerator+1) / (denominator+1))

def weighted_ce_loss(y_pred, y_true, alpha=64, smooth=1):
    weight1 = torch.sum(y_true==1,dim=(-1,-2))+smooth
    weight0 = torch.sum(y_true==0, dim=(-1,-2))+smooth
    multiplier_1 = weight0/(weight1*alpha)
    multiplier_1 = multiplier_1.view(-1,1,1)
    # print(multiplier_1.shape)
    # print(y_pred.shape)
    # print(y_true.shape)

    loss = -torch.mean(torch.mean((multiplier_1*y_true*torch.log(y_pred)) + (1-y_true)*(torch.log(1-y_pred)),dim=(-1,-2)))
    return loss

def focal_loss(y_pred, y_true, alpha_def=0.75, gamma=3):
    # print('going back to the default value of alpha')
    alpha = alpha_def
    ce_loss = F.binary_cross_entropy_with_logits(y_pred, y_true, reduction="none")
    assert (ce_loss>=0).all()
    p_t = y_pred * y_true + (1 - y_pred) * (1 - y_true)
    # 1/0
    loss = ce_loss * ((1 - p_t) ** gamma)
    alpha_t = alpha * y_true + (1 - alpha) * (1 - y_true)
    loss = alpha_t * loss
    loss = torch.mean(loss, dim=(-1,-2))
    return loss.mean()

def multiclass_focal_loss(y_pred, y_true, alpha = 0.75, gamma=3):
    if len(y_pred.shape)==4:
        y_pred = y_pred.squeeze()
    ce = y_true*(-torch.log(y_pred))
    weight = y_true * ((1-y_pred)**gamma)
    fl = torch.sum(alpha*weight*ce, dim=(-1,-2))
    return torch.mean(fl)

def str2bool(v):
    if v.lower() in ['true', 1]:
        return True
    elif v.lower() in ['false', 0]:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class AverageMeter(object):
    """Computes and stores the average and current value"""

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

class qkv_transform(nn.Conv1d):
    """Conv1d for qkv_transform"""
