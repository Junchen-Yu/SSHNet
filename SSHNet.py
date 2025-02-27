import torch.nn as nn
from utils.utils import *
import importlib
import network.modal_trans.trans as modal_trans
import network.homo_estimator.estimator as homo_estimator

class SSHNet(nn.Module):
    def __init__(self, trans="TransformerUNet", homo="IHN"):
        super().__init__()
        
        if trans == "None":
            self.modal_trans = None
        else:
            self.modal_trans = getattr(modal_trans, trans)()       
        self.homo_estimator = getattr(homo_estimator, homo)()

    def forward(self, image1, image2, mode):
        if mode == "train11":
            if self.modal_trans != None:
                pseudo_patch_img1_warp, pseudo_patch_img1 = self.modal_trans(image1), self.modal_trans(image2)
                pred_h4p_11 = self.homo_estimator(pseudo_patch_img1_warp.detach(), pseudo_patch_img1.detach())
            else:
                pred_h4p_11 = self.homo_estimator(image1, image2)
            return pred_h4p_11
        elif mode == "train22":
            pred_h4p_22 = self.homo_estimator(image1, image2)
            return pred_h4p_22
        elif mode == "train12" or mode == "test":
            if self.modal_trans != None:
                pseudo_patch_img1 = self.modal_trans(image1)
                pred_h4p_12 = self.homo_estimator(pseudo_patch_img1.detach(), image2)
            else:
                pseudo_patch_img1 = image1
                pred_h4p_12 = self.homo_estimator(image1, image2)
            return pseudo_patch_img1, pred_h4p_12
        else:
            print("ERROR : mode error")