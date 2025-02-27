import torch
import torch.nn as nn
import torch.nn.functional as F
import lpips
from einops import rearrange

class CorrLoss(torch.nn.Module):
    def __init__(self, pool = False):
        super(CorrLoss, self).__init__()
        self.pool = pool

    def forward(self, fmap1, fmap2):
        if self.pool:
            fmap1 = F.avg_pool2d(fmap1, kernel_size=2, stride=2)
            fmap2 = F.avg_pool2d(fmap2, kernel_size=2, stride=2)
        fmap1 = rearrange(fmap1, "b c h w -> b (h w) c").contiguous()
        fmap2 = rearrange(fmap2, "b c h w -> b (h w) c").contiguous()
        f_corr_sim = nn.CosineSimilarity(dim=2).cuda(fmap1.get_device())
        f_similarity = f_corr_sim(fmap1, fmap2).mean()
        loss = 1 - f_similarity
        return loss

def loss_func(loss_type):
    if loss_type == "perceptual": 
        loss = lpips.LPIPS(net='vgg')
    elif loss_type == "l1": 
        loss = nn.L1Loss()
    elif loss_type == "l2": 
        loss = nn.MSELoss()
    elif loss_type == "corr":
        loss = CorrLoss()
    elif loss_type == "pool_corr": 
        loss = CorrLoss(pool=True)
    elif loss_type == "None":
        loss = None
    
    return loss