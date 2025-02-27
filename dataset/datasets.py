import numpy as np
from torch.utils.data import Dataset, DataLoader
from glob import glob
import os
import cv2
import albumentations as A
from utils.homo_utils import generate_homo

class homo_dataset(Dataset):
    def __init__(self, split, dataset, args):
        self.dataset = dataset
        self.args = args
        self.homo_parameter = {"marginal":32, "perturb":32, "patch_size":128}
        self.transform = A.Compose([A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0, brightness_by_max=False, always_apply=True),
                                    A.Sharpen(always_apply=True)])
        self.augmentation = A.ReplayCompose([A.VerticalFlip(always_apply = False,p = 0.5),
                                             A.HorizontalFlip(always_apply = False,p = 0.5),
                                             A.Rotate(limit=90, interpolation=1, border_mode=4, value=None, mask_value=None, always_apply=False, p=0.5)])
        
        if split == 'train':
            if dataset == 'ggmap':
                root_img1 = '/home/root/Homography/datasets/GoogleMap/train2014_input'
                root_img2 = '/home/root/Homography/datasets/GoogleMap/train2014_template_original'
            if dataset == 'optsar':
                root_img1 = '/home/root/Homography/datasets/optsar/train/opt'
                root_img2 = '/home/root/Homography/datasets/optsar/train/sar'   
            if dataset == 'dpdn':
                root_img1 = '/home/root/Homography/datasets/DPDN_192/train/rgb'
                root_img2 = '/home/root/Homography/datasets/DPDN_192/train/depth'
            if dataset == "fnf":
                root_img1 = '/home/root/Homography/datasets/Flash_no_flash_resized/trainA'
                root_img2 = '/home/root/Homography/datasets/Flash_no_flash_resized/trainB'
            if dataset == "rgbnir":
                root_img1 = '/home/root/Homography/datasets/RGBNIR/rgb'
                root_img2 = '/home/root/Homography/datasets/RGBNIR/nir'
        else:
            if dataset == 'ggmap':
                root_img1 = '/home/root/Homography/datasets/GoogleMap/val2014_input'
                root_img2 = '/home/root/Homography/datasets/GoogleMap/val2014_template_original'
            if dataset == 'optsar':
                root_img1 = '/home/root/Homography/datasets/optsar/test/opt'
                root_img2 = '/home/root/Homography/datasets/optsar/test/sar' 
            if dataset == 'dpdn':
                root_img1 = '/home/root/Homography/datasets/DPDN_192/test/rgb'
                root_img2 = '/home/root/Homography/datasets/DPDN_192/test/depth'
            if dataset == "fnf":
                root_img1 = '/home/root/Homography/datasets/Flash_no_flash_resized/testA'
                root_img2 = '/home/root/Homography/datasets/Flash_no_flash_resized/testB'
            if dataset == "rgbnir":
                root_img1 = '/home/root/Homography/datasets/RGBNIR/rgb'
                root_img2 = '/home/root/Homography/datasets/RGBNIR/nir'

        if dataset == "rgbnir":
            self.image_list_img1 = sorted(glob(os.path.join(root_img1, '*.bmp')))
            self.image_list_img2 = sorted(glob(os.path.join(root_img2, '*.bmp')))
        else:
            self.image_list_img1 = sorted(glob(os.path.join(root_img1, '*.jpg')))
            self.image_list_img2 = sorted(glob(os.path.join(root_img2, '*.jpg')))
                

    def __len__(self):
        return len(self.image_list_img1)

    def __getitem__(self, index):
        img1 = cv2.imread(self.image_list_img1[index])
        img2 = cv2.imread(self.image_list_img2[index])
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        
        if self.dataset in ['optsar']:
            img_size = self.homo_parameter["patch_size"] + 2 * self.homo_parameter["marginal"]
            img1 = cv2.resize(img1, (img_size, img_size))
            img2 = cv2.resize(img2, (img_size, img_size))
        
        if self.dataset in ['rgbnir']:
            img1 = cv2.resize(img1, (256, 256))
            img2 = cv2.resize(img2, (256, 256))

        if self.dataset in ['fnf','rgbnir']:
            augmented = self.augmentation(image=img1)
            img1 = augmented['image']
            replay = augmented['replay']
            img2 = A.ReplayCompose.replay(replay, image=img2)['image']       

        self.homo_parameter["height"], self.homo_parameter["width"], _ = img1.shape
        
        if self.args.randomize:
            patch_img1_warp, patch_img1, gt11, _, _, _, _ = generate_homo(img1, img1, homo_parameter=self.homo_parameter, transform=self.transform)
            patch_img2_warp, patch_img2, gt22, _, _, _, _ = generate_homo(img2, img2, homo_parameter=self.homo_parameter, transform=self.transform)
        else:
            patch_img1_warp, patch_img1, gt11, _, _, _, _ = generate_homo(img1, img1, homo_parameter=self.homo_parameter, transform=None)
            patch_img2_warp, patch_img2, gt22, _, _, _, _ = generate_homo(img2, img2, homo_parameter=self.homo_parameter, transform=None)
        patch_img1_eval, patch_img2_eval, gt12_eval, org_pts_eval, dst_pts_eval, large_img1_eval, large_img2_eval = generate_homo(img1, img2, homo_parameter=self.homo_parameter, transform=None)
        
        return {"patch_img1_warp":patch_img1_warp, "patch_img1":patch_img1, "gt11":gt11, 
                "patch_img2_warp":patch_img2_warp, "patch_img2":patch_img2, "gt22":gt22, 
                "patch_img1_eval":patch_img1_eval, "patch_img2_eval":patch_img2_eval, "gt12_eval":gt12_eval,
                "org_pts_eval":org_pts_eval, "dst_pts_eval":dst_pts_eval,
                "large_img1_eval":large_img1_eval, "large_img2_eval":large_img2_eval}

def fetch_dataloader(args, split='train'):
    if split == 'train':
        train_dataset = homo_dataset(split='train', dataset=args.dataset, args=args)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True, shuffle=True, num_workers=16, drop_last=False)
        print('Training with %d image pairs' % len(train_dataset))
    else: 
        train_dataset = homo_dataset(split='test', dataset=args.dataset, args=args)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True, shuffle=True, num_workers=16, drop_last=False)
        print('Test with %d image pairs' % len(train_dataset))
    
    return train_loader


