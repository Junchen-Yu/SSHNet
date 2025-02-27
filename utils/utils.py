import os, random, sys
import numpy as np
import torch
import torchvision
import cv2
import matplotlib.pyplot as plt
from kornia.geometry import get_perspective_transform, warp_perspective

class Logger_(object):
    def __init__(self, filename=None, stream=sys.stdout):
            self.terminal = stream
            self.log = open(filename, 'a')

    def write(self, message):
            self.terminal.write(message)
            self.log.write(message)

    def flush(self):
            pass

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def visualize_input_image(args, data_batch):
    visualize_path = os.path.join(args.log_full_dir, "visualize", "input")
    if not os.path.exists(visualize_path): os.makedirs(visualize_path)
    patch_img1_warp, patch_img1 = data_batch["patch_img1_warp"], data_batch["patch_img1"]
    patch_img2_warp, patch_img2 = data_batch["patch_img2_warp"], data_batch["patch_img2"]
    patch_img1_eval, patch_img2_eval = data_batch["patch_img1_eval"], data_batch["patch_img2_eval"]
    large_img1_eval, large_img2_eval = data_batch["large_img1_eval"], data_batch["large_img2_eval"]
    
    torchvision.utils.save_image(patch_img1_warp, os.path.join(visualize_path, 'patch_img1_warp.jpg'))
    torchvision.utils.save_image(patch_img1, os.path.join(visualize_path, 'patch_img1.jpg'))
    torchvision.utils.save_image(patch_img2_warp, os.path.join(visualize_path, 'patch_img2_warp.jpg'))
    torchvision.utils.save_image(patch_img2, os.path.join(visualize_path, 'patch_img2.jpg'))
    torchvision.utils.save_image(patch_img1_eval, os.path.join(visualize_path, 'patch_img1_eval.jpg'))
    torchvision.utils.save_image(patch_img2_eval, os.path.join(visualize_path, 'patch_img2_eval.jpg'))
    torchvision.utils.save_image(large_img1_eval, os.path.join(visualize_path, 'large_img1_eval.jpg'))
    torchvision.utils.save_image(large_img2_eval, os.path.join(visualize_path, 'large_img2_eval.jpg'))
    
    org_pts = data_batch["org_pts_eval"]
    org_pts = org_pts - org_pts[:, [0]]
    dst_pts = org_pts + data_batch["gt12_eval"]
    H_gt = get_perspective_transform(org_pts, dst_pts)
    patch_size = data_batch["patch_img1_eval"].shape[2]
    patch_img1_eval_check = warp_perspective(data_batch["patch_img1_eval"], H_gt, (patch_size, patch_size))
    torchvision.utils.save_image(patch_img1_eval_check, os.path.join(visualize_path, 'patch_img1_eval_check.jpg'))

def visualize_predict_image(args, data_batch, pred_h4p_12, iters, model_flag=None):
    # model_flag ["teacher", "student"]
    if model_flag is not None: visualize_path = os.path.join(args.log_full_dir, "visualize", "output", "test", model_flag)
    else: visualize_path = os.path.join(args.log_full_dir, "visualize", "output", "test")
    if not os.path.exists(visualize_path): os.makedirs(visualize_path)
    
    pred_h4p_12 = pred_h4p_12[-1]
    org_pts_eval = data_batch["org_pts_eval"]
    dst_pts_eval = data_batch["dst_pts_eval"]
    dst_pts_pred = data_batch["org_pts_eval"] + pred_h4p_12
    H = get_perspective_transform(org_pts_eval, dst_pts_pred).double()
    large_size, patch_size = data_batch["large_img1_eval"].shape[2], data_batch["patch_img1_eval"].shape[2]
    large_img1_eval_predict = warp_perspective(data_batch["large_img1_eval"], H, dsize=(large_size, large_size))
    torchvision.utils.save_image(large_img1_eval_predict, os.path.join(visualize_path, f'{iters}_large_img1_eval_predict.jpg'))
    
    patch_img1_eval_predict = torch.zeros((pred_h4p_12.shape[0], 3, patch_size, patch_size)).to(pred_h4p_12.device)
    top_left, bottom_right = [], []
    for idx in range(pred_h4p_12.shape[0]):
        top_left = org_pts_eval[idx][0].cpu().numpy().astype(np.int32)
        bottom_right = org_pts_eval[idx][3].cpu().numpy().astype(np.int32)
        patch_img1_eval_predict[idx] = large_img1_eval_predict[idx, :, top_left[1]:bottom_right[1]+1, top_left[0]:bottom_right[0]+1]
    torchvision.utils.save_image(patch_img1_eval_predict, os.path.join(visualize_path, f'{iters}_patch_img1_eval_predict.jpg'))
    torchvision.utils.save_image(data_batch["patch_img1_eval"], os.path.join(visualize_path, f'{iters}_patch_img1_eval.jpg'))
    torchvision.utils.save_image(data_batch["patch_img2_eval"], os.path.join(visualize_path, f'{iters}_patch_img2_eval.jpg'))
    
    org_pts_eval = org_pts_eval.reshape(-1, 4, 1, 2).cpu().numpy()[:,[0,2,3,1]].astype(np.int32)
    dst_pts_eval = dst_pts_eval.reshape(-1, 4, 1, 2).cpu().numpy()[:,[0,2,3,1]].astype(np.int32)
    dst_pts_pred = org_pts_eval + pred_h4p_12.reshape(-1, 4, 1, 2).detach().cpu().numpy()[:,[0,2,3,1]].astype(np.int32)
    
    large_img1_eval = data_batch["large_img1_eval"].permute(0,2,3,1).cpu().numpy().copy()
    large_img2_eval = data_batch["large_img2_eval"].permute(0,2,3,1).cpu().numpy().copy()
    
    for idx in range(pred_h4p_12.shape[0]):
        large_img1_eval[idx] = cv2.polylines(large_img1_eval[idx], np.int32([org_pts_eval[idx]]), True, (0,1,0), 3, cv2.LINE_AA)
        large_img2_eval[idx] = cv2.polylines(large_img2_eval[idx], np.int32([dst_pts_eval[idx]]), True, (0,1,0), 3, cv2.LINE_AA)
        large_img2_eval[idx] = cv2.polylines(large_img2_eval[idx], np.int32([dst_pts_pred[idx]]), True, (1,0,0), 2, cv2.LINE_AA)

    torchvision.utils.save_image(torch.from_numpy(large_img1_eval).permute(0,3,1,2), 
                                 os.path.join(visualize_path, f'{iters}_large_img1_eval_draw.jpg'))
    torchvision.utils.save_image(torch.from_numpy(large_img2_eval).permute(0,3,1,2), 
                                 os.path.join(visualize_path, f'{iters}_large_img2_eval_draw.jpg'))
    
def visualize_eval_image(args, data_batch, pred_h4p_12, iters, model_flag=None):
    # model_flag ["teacher", "student"]
    if model_flag is not None: visualize_path = os.path.join(args.log_full_dir, "visualize", model_flag)
    else: visualize_path = os.path.join(args.log_full_dir, "visualize")
    if not os.path.exists(visualize_path): os.makedirs(visualize_path)
    
    pred_h4p_12 = pred_h4p_12[-1]
    org_pts_eval = data_batch["org_pts_eval"]
    dst_pts_eval = data_batch["dst_pts_eval"]
    dst_pts_pred = data_batch["org_pts_eval"] + pred_h4p_12
    H = get_perspective_transform(org_pts_eval, dst_pts_pred).double()
    large_size, patch_size = data_batch["large_img1_eval"].shape[2], data_batch["patch_img1_eval"].shape[2]
    large_img1_eval_predict = warp_perspective(data_batch["large_img1_eval"], H, dsize=(large_size, large_size))
   
    patch_img1_eval_predict = torch.zeros((pred_h4p_12.shape[0], 3, patch_size, patch_size)).to(pred_h4p_12.device)
    top_left, bottom_right = [], []
    for idx in range(pred_h4p_12.shape[0]):
        top_left = org_pts_eval[idx][0].cpu().numpy().astype(np.int32)
        bottom_right = org_pts_eval[idx][3].cpu().numpy().astype(np.int32)
        patch_img1_eval_predict[idx] = large_img1_eval_predict[idx, :, top_left[1]:bottom_right[1]+1, top_left[0]:bottom_right[0]+1]
        
    org_pts_eval = org_pts_eval.reshape(-1, 4, 1, 2).cpu().numpy()[:,[0,2,3,1]].astype(np.int32)
    dst_pts_eval = dst_pts_eval.reshape(-1, 4, 1, 2).cpu().numpy()[:,[0,2,3,1]].astype(np.int32)
    dst_pts_pred = org_pts_eval + pred_h4p_12.reshape(-1, 4, 1, 2).detach().cpu().numpy()[:,[0,2,3,1]].astype(np.int32)
    
    large_img1_eval = data_batch["large_img1_eval"].permute(0,2,3,1).cpu().numpy().copy()
    large_img2_eval = data_batch["large_img2_eval"].permute(0,2,3,1).cpu().numpy().copy()
    
    for idx in range(pred_h4p_12.shape[0]):
        large_img1_eval[idx] = cv2.polylines(large_img1_eval[idx], np.int32([org_pts_eval[idx]]), True, (0,1,0), 3, cv2.LINE_AA)
        large_img2_eval[idx] = cv2.polylines(large_img2_eval[idx], np.int32([dst_pts_eval[idx]]), True, (0,1,0), 3, cv2.LINE_AA)
        large_img2_eval[idx] = cv2.polylines(large_img2_eval[idx], np.int32([dst_pts_pred[idx]]), True, (1,0,0), 2, cv2.LINE_AA)
        
    torchvision.utils.save_image(torch.from_numpy(large_img2_eval).permute(0,3,1,2), 
                                 os.path.join(visualize_path, str(iters).zfill(3) + "_eval_draw.jpg"))

def visualize_mace(args, data_batch, pred_h4p_12, iters, model_flag=None):
    if model_flag is not None: visualize_path = os.path.join(args.log_full_dir, "visualize", model_flag)
    else: visualize_path = os.path.join(args.log_full_dir, "visualize")
    if not os.path.exists(visualize_path): os.makedirs(visualize_path)
    mace = [((pred_h4p_12[i] - data_batch["gt12_eval"])**2).sum(dim=-1).sqrt().mean(dim=-1).detach().cpu().numpy() for i in range(len(pred_h4p_12))]
    mace = np.array(mace)
    
    num_batches = mace.shape[1]
    num_iter = mace.shape[0]
    fig, axs = plt.subplots(2, int(num_batches/2), figsize=(40, 10))
    
    for i in range(num_batches):
        row = i // int(num_batches/2)
        col = i % int(num_batches/2)
        axs[row, col].plot(np.arange(1, num_iter + 1), mace[:,i], marker='o', linestyle='-')
        if col == 0:
            axs[row, col].set_ylabel('MACE', fontsize=20)
        axs[row, col].tick_params(axis='both', which='major', labelsize=15)
            
    plt.suptitle('MACE During Iterations', fontsize=40)    
    plt.savefig(os.path.join(visualize_path, str(iters).zfill(3) + "_iter_mace.jpg"))
    
def visualize_translation_image(args, data_batch, pseudo_patch_img1_eval, pred_h4p_12, iters, mode="train", model_flag=None):
    # model_flag ["teacher", "student"]
    if model_flag is not None: visualize_path = os.path.join(args.log_full_dir, "visualize", "output", mode, model_flag)
    else: visualize_path = os.path.join(args.log_full_dir, "visualize", "output", mode)
    if not os.path.exists(visualize_path): os.makedirs(visualize_path)
    
    org_pts = data_batch["org_pts_eval"]
    org_pts = org_pts - org_pts[:, [0]]
    dst_pts = org_pts + pred_h4p_12[-1]
    H_pred = get_perspective_transform(org_pts, dst_pts)
    patch_size = data_batch["patch_img1_eval"].shape[2]
    patch_img1_eval_predict = warp_perspective(data_batch["patch_img1_eval"], H_pred, (patch_size, patch_size))
    pseudo_patch_img1_eval_predict = warp_perspective(pseudo_patch_img1_eval, H_pred, (patch_size, patch_size))
    
    torchvision.utils.save_image(patch_img1_eval_predict, os.path.join(visualize_path, f'trans_{iters}_patch_img1_eval.jpg'))
    torchvision.utils.save_image(pseudo_patch_img1_eval_predict, os.path.join(visualize_path, f'trans_{iters}_pseudo_patch_img1_eval.jpg'))
    torchvision.utils.save_image(data_batch["patch_img2_eval"], os.path.join(visualize_path, f'trans_{iters}_patch_img2_eval.jpg'))
    
def plot_result(args, plot_mace12, name):
    plt.clf()
    x_data = np.arange(0.01, 100, 0.01)
    y_data = []
    for x in x_data:
        y_data.append((plot_mace12 < x).sum() / plot_mace12.shape[0])
    plt.plot(x_data, y_data)
    plt.xscale('log')
    plt.xlim((0.1, 100))
    plt.ylim((0, 1))
    plt.grid(which='major', axis='both')
    plt.grid(which='minor', axis='x')
    plt.savefig(os.path.join(args.log_full_dir, "visualize", name + "_ace.png"))
    
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)