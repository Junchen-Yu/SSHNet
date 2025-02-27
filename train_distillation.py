import os, time, pprint
import argparse
import warnings
warnings.filterwarnings("ignore")
import torch
from torch import optim
import dataset.datasets as datasets
from utils.homo_utils import *
from SSHNet import *
from utils.utils import *
torch.backends.cuda.preferred_linalg_library("cusolver")

def train(args):
    device = torch.device("cuda:"+ str(args.gpuid))
    train_loader = datasets.fetch_dataloader(args, split="train")

    teacher_model = SSHNet(trans=args.trans, homo=args.homo).to(device)
    if args.checkpoint is None:
        print("ERROR : no checkpoint")
        exit()
    state = torch.load(args.checkpoint, map_location=device)
    teacher_model.load_state_dict(state["SSHNet"])
    print("load teacher model")
    teacher_model.eval()

    student_model = SSHNet(trans="None", homo=args.homo).to(device)
    student_model.train()

    print(f"{round(count_parameters(student_model)/1000000, 2)}M parameters")
    optimizer = optim.AdamW(list(student_model.parameters()), lr=args.lr, weight_decay=0.00001)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer, max_lr=args.lr, total_steps=args.num_steps+100,
                                              pct_start=0.05, cycle_momentum=False, anneal_strategy="linear")
        
    teacher_sum_ace12, student_sum_ace12, glob_iter = 0, 0, 0
    start_time = time.time()

    while glob_iter <= args.num_steps:
        for i, data_batch in enumerate(train_loader):
            end_time = time.time() # calculate time remaining
            
            if glob_iter == 0 and not args.nolog: visualize_input_image(args, data_batch)
            for key, value in data_batch.items(): data_batch[key] = data_batch[key].to(device)
            
            optimizer.zero_grad()

            with torch.no_grad():
                _, teacher_pred_h4p_12 = teacher_model(data_batch["patch_img1_eval"], data_batch["patch_img2_eval"], "test")

            # distillation student model
            _, student_pred_h4p_12 = student_model(data_batch["patch_img1_eval"], data_batch["patch_img2_eval"], "train12")
            distillation_loss = sequence_loss(student_pred_h4p_12, teacher_pred_h4p_12[-1].detach())

            teacher_ace12 = calculate_ace(teacher_pred_h4p_12, data_batch["gt12_eval"])
            student_ace12 = calculate_ace(student_pred_h4p_12, data_batch["gt12_eval"])

            loss = distillation_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), 1)
            optimizer.step()
            scheduler.step()
            
            # calculate metric
            teacher_sum_ace12 += teacher_ace12
            student_sum_ace12 += student_ace12
            
            if glob_iter % args.print_freq == 0 and glob_iter != 0:
                time_remain = (end_time - start_time) * (args.num_steps - glob_iter) / glob_iter
                print("Training: Iter[{:0>3}]/[{:0>3}] teacher_mace12: {:.3f} student_mace12: {:.3f} lr={:.8f} time: {:.2f}h".format(glob_iter, args.num_steps, 
                                                                                                                                     teacher_sum_ace12 / args.print_freq,  
                                                                                                                                     student_sum_ace12 / args.print_freq,
                                                                                                                                     scheduler.get_lr()[0], time_remain/3600))
                
                teacher_sum_ace12, student_sum_ace12 = 0, 0

            # save model
            if glob_iter % args.save_freq == 0 and glob_iter != 0 and not args.nolog:
                filename = "model" + "_iter_" + str(glob_iter) + ".pth"
                model_save_path = os.path.join(args.log_full_dir, filename)
                checkpoint = {"SSHNet": teacher_model.state_dict(), "SSHNet-D": student_model.state_dict()}
                torch.save(checkpoint, model_save_path)
                args.checkpoint = model_save_path
            
            if glob_iter % args.val_freq == 0 and glob_iter != 0:
                test(args, glob_iter, teacher_model, student_model)

            glob_iter += 1
            if glob_iter > args.num_steps: break

            
def test(args, glob_iter=None, teacher_model=None, student_model=None):
    device = torch.device("cuda:"+ str(args.gpuid))
    test_loader = datasets.fetch_dataloader(args, split="test")
    if teacher_model == None:
        teacher_model = SSHNet(trans=args.trans, homo=args.homo).to(device)
        student_model = SSHNet(trans="None", homo=args.homo).to(device)
        if args.checkpoint is None:
            print("ERROR : no checkpoint")
            exit()
        state = torch.load(args.checkpoint, map_location=device)
        teacher_model.load_state_dict(state["SSHNet"])
        student_model.load_state_dict(state["SSHNet-D"])
        print("test with pretrained model")
    teacher_model.eval()
    student_model.eval()

    with torch.no_grad():
        teacher_sum_ace12, studen_sum_ace12, batch_num = 0, 0, 0
        for test_repeat in range(5): # repeat test multiple times to get stable result
            for i, data_batch in enumerate(test_loader):
                for key, value in data_batch.items(): 
                    if type(data_batch[key]) == torch.Tensor: data_batch[key] = data_batch[key].to(device)
                    
                pseudo_patch_img1_eval, teacher_pred_h4p_12 = teacher_model(data_batch["patch_img1_eval"], data_batch["patch_img2_eval"], "test")
                _, student_pred_h4p_12 = student_model(data_batch["patch_img1_eval"], data_batch["patch_img2_eval"], "test")
                
                # calculate metric
                teacher_ace12 = calculate_ace(teacher_pred_h4p_12, data_batch["gt12_eval"])
                student_ace12 = calculate_ace(student_pred_h4p_12, data_batch["gt12_eval"])
                teacher_sum_ace12 += teacher_ace12
                studen_sum_ace12 += student_ace12
                batch_num += 1
    
    visualize_predict_image(args, data_batch, student_pred_h4p_12, glob_iter, "student")
    visualize_translation_image(args, data_batch, pseudo_patch_img1_eval, student_pred_h4p_12, glob_iter, "test", "student")
    print(f"total_batch:{batch_num} teacher_mace12:{round(teacher_sum_ace12 / batch_num, 3)} student_mace12:{round(studen_sum_ace12 / batch_num, 3)}")

    student_model.train()
            
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train", help="Train or test", choices=["train", "test"])
    parser.add_argument("--trans", type=str, default="TransformerUNet", help="modality transfer network")
    parser.add_argument("--homo", type=str, default="IHN", help="homography estimation network")
    parser.add_argument("--gpuid", type=int, default=1)
    parser.add_argument("--note", type=str, default="", help="experiment notes")
    parser.add_argument("--dataset", type=str, default="ggmap", help="dataset")
    parser.add_argument("--log_dir", type=str, default="logs", help="The log path")
    parser.add_argument("--nolog", action="store_true", default=False, help="save log file or not")
    parser.add_argument("--checkpoint", type=str, help="Test model name")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--save_freq", type=int, default=120000)
    parser.add_argument("--print_freq", type=int, default=100)
    parser.add_argument("--val_freq", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--num_steps", type=int, default=120000)
    parser.add_argument("--lr", type=float, default=4e-4, help="Max learning rate")
    parser.add_argument("--randomize", type=bool, default=True)
    parser.add_argument("--no-randomize", dest="randomize", action="store_false", help="Whether to randomize")
    parser.add_argument("--log_full_dir", type=str)
    parser.add_argument("--loss_type", type=str, default="perceptual")
    parser.add_argument("--loss_type_f", type=str, default="cos")
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
