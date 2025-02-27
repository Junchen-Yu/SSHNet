import torch
import torch.nn as nn
from network.homo_estimator.update import *
from network.homo_estimator.extractor import *
from network.homo_estimator.corr import *
from network.homo_estimator.utils import *
from network.homo_estimator.ATT.attention_layer import Correlation, FocusFormer_Attention
from utils.flow_utils import *

class IHN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fnet = BasicEncoder(output_dim=256, norm_fn='instance')
        self.update_block = GMA()

    def forward(self, image1, image2, iters = 6):
        fmap1 = self.fnet(image1)
        fmap2 = self.fnet(image2)
        batch_size = fmap1.shape[0]

        corr_fn = CorrBlock(fmap1, fmap2, num_levels=2, radius=4, sz=32)
        coords0, coords1 = initialize_flow(image1, downsample=4)
        four_point_disp = torch.zeros((batch_size, 2, 2, 2)).to(fmap1.device)
        four_point_predictions = []

        for itr in range(iters):
            coords1 = disp_to_flow(four_point_disp, coords0, downsample=4)
            corr = corr_fn(coords1.detach())
            flow = coords1 - coords0
            delta_four_point = self.update_block(corr, flow)
            four_point_disp =  four_point_disp + delta_four_point
            four_point_reshape = four_point_disp.permute(0,2,3,1).reshape(-1,4,2) # [top_left, top_right, bottom_left, bottom_right], [-1, 4, 2]
            four_point_predictions.append(four_point_reshape)
        
        return four_point_predictions

class RHWF(nn.Module):
    def __init__(self):
        super().__init__()
        self.fnet = RHWF_Encoder(output_dim=256, norm_fn='instance')
        self.conv3 = Conv3(input_dim=130)

        self.lev0 = True
        self.lev1 = False
        
        self.transformer_0 = FocusFormer_Attention(256, 1, 96, 96)
        self.kernel_list_0 = [0, 9, 5, 3, 3, 3]
        self.pad_list_0    = [0, 4, 2, 1, 1, 1]
        sz = 32
        self.kernel_0 = 17
        self.pad_0 = 8
        self.conv1_0 = Conv1(input_dim=145)
        self.update_block_4 = GMA_update(sz)
        
    def forward(self, image1, image2, iters = 6):        
        image1 = image1.contiguous()
        image2 = image2.contiguous()
        image2_org = image2

        # feature network
        fmap1_32 = self.fnet(image1)
        fmap2_32 = self.fnet(image2)
        
        four_point_disp = torch.zeros((image1.shape[0], 2, 2, 2)).to(image1.device)
        four_point_predictions = []

        coords0, coords1 = initialize_flow(image1, 4)
        coords0 = coords0.detach()
        sz = fmap1_32.shape
        self.sz = sz
        self.get_flow_now_4 = Get_Flow(sz)
        
        for itr in range(iters):
            if itr < 6:
                fmap1, fmap2 = self.transformer_0(fmap1_32, fmap2_32, self.kernel_list_0[itr], self.pad_list_0[itr])
            else:
                fmap1, fmap2 = self.transformer_0(fmap1_32, fmap2_32, 3, 1)
                
            coords1 = coords1.detach()
            corr = F.relu(Correlation.apply(fmap1.contiguous(), fmap2.contiguous(), self.kernel_0, self.pad_0)) 
            b, h, w, _ = corr.shape
            corr_1 = F.avg_pool2d(corr.view(b, h*w, self.kernel_0, self.kernel_0), 2).view(b, h, w, 64).permute(0, 3, 1, 2)
            corr_2 = corr.view(b, h*w, self.kernel_0, self.kernel_0)
            corr_2 = corr_2[:,:,4:13,4:13].contiguous().view(b, h, w, 81).permute(0, 3, 1, 2)                                                  
            corr = torch.cat([corr_1, corr_2], dim=1)

            corr = self.conv1_0(corr)                                         
            flow = coords1 - coords0
            corr_flow = torch.cat((corr, flow), dim=1)
            corr_flow = self.conv3(corr_flow)             
            
            delta_four_point = self.update_block_4(corr_flow)
            four_point_disp =  four_point_disp + delta_four_point
            four_point_reshape = four_point_disp.permute(0,2,3,1).reshape(-1,4,2)
            four_point_predictions.append(four_point_reshape)
            coords1 = self.get_flow_now_4(four_point_disp, 4)
            
            if itr < (iters-1):
                flow_med = coords1 - coords0
                flow_med = F.upsample_bilinear(flow_med, None, [4, 4]) * 4              
                flow_med = flow_med.detach()         
                image2_warp = warp(image2_org, flow_med)
                fmap2_32_warp  = self.fnet(image2_warp)
                fmap2_32 = fmap2_32_warp.float()               

        return four_point_predictions
    
class MHN(nn.Module):
    def __init__(self):
        super(MHN, self).__init__()
        self.net2 = Net2()
        self.net1 = Net1()
        self.net0 = Net0()
        for model in [self.net2, self.net1, self.net0]: model.apply(weights_init)
        
    def forward(self, image1, image2):
        patch_size = image1.shape[-1]
        batch_size = image1.shape[0]
        four_pts_org = torch.tensor([(0, 0), (patch_size - 1, 0), (0, patch_size - 1), (patch_size - 1, patch_size - 1)], dtype=torch.float32).to(image1.device)
        four_pts_org = four_pts_org.unsqueeze(0).expand(batch_size, -1, -1)

        img1, img2 = image1.mean(1, keepdims=True), image2.mean(1, keepdims=True)
        img1_pyramid, img2_pyramid = pyramid(img1), pyramid(img2)
        
        disp2 = self.net2(torch.cat([img1_pyramid[0], img2_pyramid[0]], dim=1))
        four_pts_pred = four_pts_org + disp2
        H2 = tgm.get_perspective_transform(four_pts_org, four_pts_pred)
        H2[:,0:2,2] = H2[:,0:2,2]/2
        warped_img1_64 = tgm.warp_perspective(img1_pyramid[1], H2, (64, 64)).detach()
        
        disp1 = self.net1(torch.cat([warped_img1_64, img2_pyramid[1]], dim=1))
        four_pts_pred = four_pts_pred + disp1
        H1 = tgm.get_perspective_transform(four_pts_org, four_pts_pred)
        warped_img1_128 = tgm.warp_perspective(img1_pyramid[2], H1, (128, 128)).detach()
        
        disp0  = self.net0(torch.cat([warped_img1_128, img2_pyramid[2]], dim=1))
        return [disp2, disp2.detach()+disp1, disp2.detach()+disp1.detach()+disp0]