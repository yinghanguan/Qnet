import torch
import torch.nn.functional as F
from tqdm import tqdm

from dice_loss import dice_coeff,SegmentationMetric


def eval_net(net, loader, device):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    mask_type = torch.float32 if net.n_classes == 1 else torch.long
    n_val = len(loader)  # the number of batch
    tot = 0
    PA = 0
    CPA = 0
    MPA = 0
    mIoU = 0
    metric = SegmentationMetric(2)
    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs, true_masks = batch['image'], batch['mask']
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=mask_type)

            with torch.no_grad():
                mask_pred = net(imgs)

            if net.n_classes > 1:
                tot += F.cross_entropy(mask_pred, true_masks).item()
            else:
                pred = torch.sigmoid(mask_pred)
                pred = (pred > 0.5).float()
                metric.addBatch(pred.cpu(), true_masks.cpu())
                PA = metric.pixelAccuracy()
                CPA = metric.classPixelAccuracy()
                MPA = metric.meanPixelAccuracy()
                mIoU = metric.meanIntersectionOverUnion()
                print('PA:', PA)
                print('CPA:',CPA)
                print('MPA:',MPA)
                print('mIoU:',mIoU)
            pbar.update()

    net.train()
    return tot / n_val
