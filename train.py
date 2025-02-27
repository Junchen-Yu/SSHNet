import os, time, pprint
import argparse
import warnings
warnings.filterwarnings("ignore")
import torch
from torch import optim
import dataset.datasets as datasets
from SSHNet import *
from utils.utils import *
from utils.loss import loss_func
from utils.homo_utils import *
from kornia.geometry.transform import get_perspective_transform
from torch.utils.tensorboard import SummaryWriter

torch.backends.cuda.preferred_linalg_library("cusolver")

def train(args):
    device = torch.device("cuda:"+ str(args.gpuid))
    train_loader = datasets.fetch_dataloader(args, split="train")
    model = SSHNet(trans=args.trans, homo=args.homo).to(device)
    model.train()

    print(f"{round(count_parameters(model)/1000000, 2)}M parameters")
    optimizer = optim.AdamW(list(model.parameters()), lr=args.lr, weight_decay=0.00001)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer, max_lr=args.lr, total_steps=args.num_steps+100,
                                              pct_start=0.05, cycle_momentum=False, anneal_strategy="linear")
    
    loss_function = loss_func(args.loss_type)
    if args.loss_type == "perceptual": loss_function = loss_function.to(device)
    loss_function_f = loss_func(args.loss_type_f)

    writer = SummaryWriter(args.log_full_dir)
    sum_ace11, sum_ace12, sum_ace22, glob_iter = 0, 0, 0, 0
    start_time = time.time()

    while glob_iter <= args.num_steps:
        for i, data_batch in enumerate(train_loader):
            end_time = time.time() # calculate time remaining
            
            if glob_iter == 0 and not args.nolog: visualize_input_image(args, data_batch)
            for key, value in data_batch.items(): data_batch[key] = data_batch[key].to(device)
            
            optimizer.zero_grad()

            pred_h4p_11 = model(data_batch["patch_img1_warp"], data_batch["patch_img1"], "train11")
            pred_h4p_22 = model(data_batch["patch_img2_warp"], data_batch["patch_img2"], "train22")
            pseudo_patch_img1, pred_h4p_12 = model(data_batch["patch_img1_eval"], data_batch["patch_img2_eval"], "train12")

            org_pts = data_batch["org_pts_eval"]
            org_pts = org_pts - org_pts[:,[0]]
            dst_pts = org_pts + pred_h4p_12[-1]
            H_pred = get_perspective_transform(org_pts, dst_pts)
            image_size = data_batch["patch_img1_warp"].shape[-1]
            pseudo_patch_img1_pred_warp = warp_perspective(pseudo_patch_img1, H_pred.detach(), dsize=(image_size, image_size))
            mask = torch.ones_like(pseudo_patch_img1_pred_warp).to(device)
            mask = warp_perspective(mask, H_pred.detach(), dsize=(image_size, image_size))
            trans_loss = loss_function(pseudo_patch_img1_pred_warp, data_batch["patch_img2_eval"] * mask).mean()

            if args.loss_type_f == "None": 
                homo_feature_loss = torch.tensor(0.0, device=device)
            elif args.homo in ["IHN", "RHWF", "SCPNet"]:
                homo_feature_1 = model.homo_estimator.fnet(pseudo_patch_img1_pred_warp)
                homo_feature_2 = model.homo_estimator.fnet(data_batch["patch_img2_eval"] * mask)
                homo_feature_loss = loss_function_f(homo_feature_1, homo_feature_2).mean()

            homo_loss_11 = sequence_loss(pred_h4p_11, data_batch["gt11"])
            homo_loss_22 = sequence_loss(pred_h4p_22, data_batch["gt22"])
            homo_loss = homo_loss_11 + homo_loss_22
            
            ace11 = calculate_ace(pred_h4p_11, data_batch["gt11"])
            ace22 = calculate_ace(pred_h4p_22, data_batch["gt22"])
            ace12 = calculate_ace(pred_h4p_12, data_batch["gt12_eval"])

            # train two networks
            loss = homo_loss + trans_loss + homo_feature_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            scheduler.step()
            if glob_iter % 20 == 0:
                writer.add_scalar("Loss_" + args.dataset + "_" + args.note + "/homoh_loss_11", homo_loss_11.item(), glob_iter)
                writer.add_scalar("Loss_" + args.dataset + "_" + args.note + "/homoh_loss_22", homo_loss_22.item(), glob_iter)
                writer.add_scalar("Loss_" + args.dataset + "_" + args.note + "/trans_loss", trans_loss.item(), glob_iter)
                writer.add_scalar("Loss_" + args.dataset + "_" + args.note + "/homo_feature_loss", homo_feature_loss.item(), glob_iter)

            # calculate metric
            sum_ace11 += ace11
            sum_ace12 += ace12
            sum_ace22 += ace22
            
            if glob_iter % args.print_freq == 0 and glob_iter != 0:
                time_remain = (end_time - start_time) * (args.num_steps - glob_iter) / glob_iter
                print("Training: Iter[{:0>3}]/[{:0>3}] mace12: {:.3f} mace11: {:.3f} mace22: {:.3f} lr={:.8f} time: {:.2f}h".format(glob_iter, 
                                                                                                                                    args.num_steps, 
                                                                                                                                    sum_ace12 / args.print_freq, 
                                                                                                                                    sum_ace11 / args.print_freq, 
                                                                                                                                    sum_ace22 / args.print_freq, 
                                                                                                                                    scheduler.get_lr()[0], 
                                                                                                                                    time_remain/3600))
                
                sum_ace11, sum_ace12, sum_ace22 = 0, 0, 0

            # save model
            if glob_iter % args.save_freq == 0 and glob_iter != 0 and not args.nolog:
                filename = "model" + "_iter_" + str(glob_iter) + ".pth"
                model_save_path = os.path.join(args.log_full_dir, filename)
                checkpoint = {"SSHNet": model.state_dict()}
                torch.save(checkpoint, model_save_path)
                args.checkpoint = model_save_path
            
            if glob_iter % args.val_freq == 0 and glob_iter != 0:
                test(args, glob_iter, model)
                visualize_translation_image(args, data_batch, pseudo_patch_img1, pred_h4p_12, glob_iter, "train", "teacher")

            glob_iter += 1
            if glob_iter > args.num_steps: break
    
    writer.close()

            
def test(args, glob_iter=None, model=None):
    device = torch.device("cuda:"+ str(args.gpuid))
    test_loader = datasets.fetch_dataloader(args, split="test")
    if model == None:
        model = SSHNet(unet_type=args.unet_type).to(device)
        if args.checkpoint is None:
            print("ERROR : no checkpoint")
            exit()
        state = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state["SSHNet"])
        print("test with pretrained model")
    model.eval()

    with torch.no_grad():
        sum_ace12, batch_num = 0, 0
        for test_repeat in range(5): # repeat test multiple times to get stable result
            for i, data_batch in enumerate(test_loader):
                for key, value in data_batch.items(): 
                    if type(data_batch[key]) == torch.Tensor: data_batch[key] = data_batch[key].to(device)
                    
                pseudo_patch_img1_eval, pred_h4p_12 = model(data_batch["patch_img1_eval"], data_batch["patch_img2_eval"], "test")
                
                # calculate metric
                ace12 = calculate_ace(pred_h4p_12, data_batch["gt12_eval"])
                sum_ace12 += ace12
                batch_num += 1
    
    visualize_predict_image(args, data_batch, pred_h4p_12, glob_iter, "teacher")
    visualize_translation_image(args, data_batch, pseudo_patch_img1_eval, pred_h4p_12, glob_iter, "test", "teacher")
    print(f"total_batch:{batch_num} mace12:{round(sum_ace12 / batch_num, 3)}")
    model.train()
            
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train", help="Train or test", choices=["train", "test"])
    parser.add_argument("--trans", type=str, default="TransformerUNet", help="modality transfer network")
    parser.add_argument("--homo", type=str, default="IHN", help="homography estimation network")
    parser.add_argument("--gpuid", type=int, default=0)
    parser.add_argument("--note", type=str, default="", help="experiment notes")
    parser.add_argument("--dataset", type=str, default="ggmap", help="dataset")
    parser.add_argument("--log_dir", type=str, default="logs", help="The log path")
    parser.add_argument("--nolog", action="store_true", default=False, help="save log file or not")
    parser.add_argument("--checkpoint", type=str, help="Test model name")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--save_freq", type=int, default=120000)
    parser.add_argument("--print_freq", type=int, default=100)
    parser.add_argument("--val_freq", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_steps", type=int, default=120000)
    parser.add_argument("--lr", type=float, default=4e-4, help="Max learning rate")
    parser.add_argument("--randomize", type=bool, default=True)
    parser.add_argument("--no-randomize", dest="randomize", action="store_false", help="Whether to randomize")
    parser.add_argument("--log_full_dir", type=str)
    parser.add_argument("--loss_type", type=str, default="perceptual")
    parser.add_argument("--loss_type_f", type=str, default="corr")
    args = parser.parse_args()
    
    if not args.nolog:
        args.log_full_dir = os.path.join(args.log_dir, time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()) + "_" + args.dataset + "_" + args.note)
        if not os.path.exists(args.log_full_dir): os.makedirs(args.log_full_dir)
        sys.stdout = Logger_(os.path.join(args.log_full_dir, f"record.log"), sys.stdout)
    pprint.pprint(vars(args))
    
    seed_everything(args.seed)
    
    if args.mode == "train": train(args)
    else: test(args)

if __name__ == "__main__":
    main()
