import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms as T
from tqdm import tqdm
import random

from utils.metrics import _dice_loss, evaluation, contrastive_loss
from utils.dataset_rotation_flip import LandmarkDataset
from models.D2GP_v19_galore_v4 import D2GPLand, Edge_Prototypes


def seed_everything(seed=42):   
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def adjust_learning_rate(optimizer, decay_factor):
    """
    Shrinks learning rate by a specified decay factor.

    :param optimizer: optimizer whose learning rate must be shrunk.
    :param decay_factor: factor to multiply learning rate with (should be < 1 for decay).
    """
    
    old_lr = optimizer.param_groups[0]["lr"]
    print(f"\nDECAYING learning rate from {old_lr:.6f}")
    
    for param_group in optimizer.param_groups:
        param_group["lr"] = param_group["lr"] * decay_factor
    
    new_lr = optimizer.param_groups[0]["lr"]
    print(f"New learning rate: {new_lr:.6f} (decay factor: {decay_factor})\n")

def main(save_path, args):
    min_delta = 0.00
    patience = 3
    epochs_no_improve = 0
    best_val_dice = 0

    train_transform = T.Compose([T.ToTensor(),])
    val_transform = T.Compose([T.ToTensor(),])

    train_dataset = LandmarkDataset(
        root=(args.data_path+'/Train'),
        transform=train_transform,
        mode='train'
    )

    val_dataset = LandmarkDataset(
        root=(args.data_path+'/Val/Val'),
        transform=val_transform,
        mode='val'
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    print('Training on:', torch.cuda.get_device_name(0), 'train sample size:', len(train_dataset), 'val sample size:', len(val_dataset), 'batch:', args.batch_size)

    device = torch.device("cuda")
    bce_loss = torch.nn.BCEWithLogitsLoss()
    cl_loss = contrastive_loss()
    best_dice = 0

    model = D2GPLand(1024, 1024).to(device)

    # print("model:", model)
    print("model parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    print("model trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    print("model total parameters:", sum(p.numel() for p in model.parameters()))
    print("rank:", args.rank)
    print("proj_type:", args.proj_type)

    optimizer = model.get_galore_optimizer(lr=args.lr, weight_decay=args.weight_decay, rank=args.rank)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epoch, args.decay_lr)

    for epoch in range(args.epoch):
        epoch_running_loss = 0
        epoch_seg_loss = 0

        # trainng
        model.train()
        # Enable anomaly detection
        torch.autograd.set_detect_anomaly(True)

        for batch_idx, (X_batch, depth, y_batch, *rest) in tqdm(enumerate(train_loader)):
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            depth = depth.to(device)

            output, _ = model(X_batch, depth)
            # print("output shape:", output.shape, "y_batch shape:", y_batch.shape, "unique y_batch:", torch.unique(y_batch))
            # output shape: [4, 4, 256, 256], y_batch(mask) shape: [4, 4, 1024, 1024]

            y_batch_down = F.interpolate(y_batch, size=(256, 256), mode='nearest').clone()
            # print("y_batch_down shape:", y_batch_down.shape, "unique y_batch_down:", torch.unique(y_batch_down))
            seg_loss = _dice_loss(output, y_batch_down) + bce_loss(output, y_batch_down)
            loss = seg_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_running_loss += loss.item()
            epoch_seg_loss += seg_loss.item()
            # epoch_contrastive_loss += prototype_loss.item()
            # epoch_edge_loss += edge_loss.item()

        # ===================log========================
        print('epoch [{}/{}], loss:{:.4f}'
              .format(epoch, args.epoch, epoch_running_loss / (batch_idx + 1)))
        print('epoch [{}/{}], seg loss:{:.4f}'
              .format(epoch, args.epoch, epoch_seg_loss / (batch_idx + 1)))

        # validation
        model.eval()
        # edge_prototypes_model.eval()
        validation_IOU = []
        mDice = []

        with torch.no_grad():
            for X_batch, depth, y_batch, name in tqdm(val_loader):
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                depth = depth.to(device)

                output, original_depth = model(X_batch, depth)
                output = torch.argmax(torch.softmax(output, dim=1), dim=1)
                y_batch = F.interpolate(y_batch, size=(256, 256), mode='nearest').clone()
                y_batch = torch.argmax(y_batch, dim=1)

                tmp2 = y_batch.detach().cpu().numpy()
                tmp = output.detach().cpu().numpy()
                tmp = tmp[0]
                tmp2 = tmp2[0]

                pred = np.array([tmp == i for i in range(4)]).astype(np.uint8)
                gt = np.array([tmp2 == i for i in range(4)]).astype(np.uint8)

                iou, dice = evaluation(pred[1:].flatten(), gt[1:].flatten())

                validation_IOU.append(iou)
                mDice.append(dice)

        print(np.mean(validation_IOU))
        print(np.mean(mDice))
        if np.mean(mDice) > best_dice:
            best_dice = np.mean(mDice)
            torch.save(model.state_dict(), save_path + "best_model_path_v19_adjust_lr_patience3.pth")
            print("save best model at", save_path + "best_model_path_v19_adjust_lr_patience3.pth")
        print("best dice is:{:.4f}".format(best_dice))

        # ---- Learning Rate adjustment logic ----
        current_dice = np.mean(mDice)
        improved = current_dice > (best_val_dice + min_delta)
        
        if improved:
            best_val_dice = current_dice
            epochs_no_improve = 0
            print(f"Validation improved! New best dice: {best_val_dice:.4f}")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epochs")
            
            if epochs_no_improve >= patience:
                print(f"Adjusting learning rate using decay_lr factor: {args.decay_lr}")
                adjust_learning_rate(optimizer, args.decay_lr)
                epochs_no_improve = 0

        # Use the cosine annealing scheduler as well
        scheduler.step()


if __name__ == '__main__':

    seed_everything(42)

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=3e-5)
    parser.add_argument('--decay_lr', type=float, default=0.8)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--num_landmark', type=int, default=3)
    parser.add_argument('--data_path', type=str, default='../../L3D_Dataset')

    # galore
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--update_proj_gap", type=int, default=50)
    parser.add_argument("--galore_scale", type=float, default=1.0)
    parser.add_argument("--proj_type", type=str, default="SRFT")
    args = parser.parse_args()

    save_path = 'results/'
    os.makedirs(save_path, exist_ok=True)


    main(save_path, args=args)
