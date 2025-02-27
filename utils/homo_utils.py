import random
import numpy as np
import cv2
import torch
import torch.nn.functional as F

def generate_homo(img1, img2, homo_parameter, transform=None):
    if transform is not None:
        img1, img2 = transform(image=img1)['image'], transform(image=img2)['image']
    img1, img2 = img1 / 255.0, img2 / 255.0 # normalize
    # define corners of image patch
    marginal, perturb, patch_size = homo_parameter["marginal"], homo_parameter["perturb"], homo_parameter["patch_size"]
    height, width = homo_parameter["height"], homo_parameter["width"]
    x = random.randint(marginal, width - marginal - patch_size)
    y = random.randint(marginal, height - marginal - patch_size)
    top_left = (x, y)
    bottom_left = (x, patch_size + y - 1)
    bottom_right = (patch_size + x - 1, patch_size + y - 1)
    top_right = (patch_size + x - 1, y)
    four_pts = np.array([top_left, top_right, bottom_left, bottom_right])
    img1 = img1[top_left[1]-marginal:bottom_right[1]+marginal+1, top_left[0]-marginal:bottom_right[0]+marginal+1, :]
    img2 = img2[top_left[1]-marginal:bottom_right[1]+marginal+1, top_left[0]-marginal:bottom_right[0]+marginal+1, :]
    four_pts = four_pts - four_pts[np.newaxis, 0] + marginal # top_left->(marginal, marginal)
    (top_left, top_right, bottom_left, bottom_right) = four_pts
    
    try:
        four_pts_perturb = []
        for i in range(4):
            t1 = random.randint(-perturb, perturb)
            t2 = random.randint(-perturb, perturb)
            four_pts_perturb.append([four_pts[i][0] + t1, four_pts[i][1] + t2])
        org_pts = np.array(four_pts, dtype=np.float32)
        dst_pts = np.array(four_pts_perturb, dtype=np.float32)
        ground_truth = dst_pts - org_pts
        H = cv2.getPerspectiveTransform(org_pts, dst_pts)
        H_inverse = np.linalg.inv(H)
    except:
        four_pts_perturb = []
        for i in range(4):
            t1 =   perturb // (i + 1)
            t2 = - perturb // (i + 1)
            four_pts_perturb.append([four_pts[i][0] + t1, four_pts[i][1] + t2])
        org_pts = np.array(four_pts, dtype=np.float32)
        dst_pts = np.array(four_pts_perturb, dtype=np.float32)
        ground_truth = dst_pts - org_pts
        H = cv2.getPerspectiveTransform(org_pts, dst_pts)
        H_inverse = np.linalg.inv(H)
    
    warped_img1 = cv2.warpPerspective(img1, H_inverse, (img1.shape[1], img1.shape[0]))
    patch_img1 = warped_img1[top_left[1]:bottom_right[1]+1, top_left[0]:bottom_right[0]+1, :]
    patch_img2 = img2[top_left[1]:bottom_right[1]+1, top_left[0]:bottom_right[0]+1, :]
    patch_img1 = torch.from_numpy(patch_img1).float().permute(2, 0, 1)
    patch_img2 = torch.from_numpy(patch_img2).float().permute(2, 0, 1)
    large_img1 = torch.from_numpy(warped_img1).permute(2, 0, 1)
    large_img2 = torch.from_numpy(img2).permute(2, 0, 1)
    
    return patch_img1, patch_img2, ground_truth, org_pts, dst_pts, large_img1, large_img2 


def sequence_loss(four_preds, flow_gt):
    """ Loss function defined over sequence of flow predictions """
    gamma = 0.8
    loss = 0

    for i in range(len(four_preds)):
        loss += gamma ** (len(four_preds) - i - 1) * F.l1_loss(four_preds[i], flow_gt)

    return loss

def calculate_ace(four_preds, flow_gt):
    
    ace = ((four_preds[-1] - flow_gt)**2).sum(dim=-1).sqrt().mean(dim=-1).detach().cpu().numpy().mean()
    
    return ace