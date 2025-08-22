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

from utils.dataset_rotation_flip_BCR import LandmarkDataset
from models.D2GP_v15_dora_bcr import D2GPLand, Edge_Prototypes


print ("v15 lora, CA v2, add rotation and flip, new loss by chat, lr: 5e-5") 

def seed_everything(seed=42):   
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def dice_loss(pred, target, epsilon=1e-6):
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2. * intersection + epsilon) / (union + epsilon)
    return 1 - dice.mean()

def proposal_induction_loss(pred_score_map, induction_map):
    bce = torch.nn.BCEWithLogitsLoss()
    return bce(pred_score_map, induction_map)

def bezier_curve(control_points, n=26):
    # control_points: (B, 6, 2)
    t = torch.linspace(0, 1, n, device=control_points.device).view(1, n, 1)  # (1, n, 1)
    # De Casteljau or Bernstein formula
    B = bernstein_poly(control_points.size(1) - 1, t)  # (1, n, 6)
    curve_points = torch.matmul(B, control_points)  # (B, n, 2)
    return curve_points

def bernstein_poly(n, t):
    # t: (1, n_pts, 1)
    coeffs = [torch.combinations(torch.tensor([i for i in range(n+1)]), r=1) for i in range(n+1)]
    from math import comb
    out = []
    for i in range(n+1):
        binom = comb(n, i)
        term = binom * (t.squeeze(-1) ** i) * ((1 - t.squeeze(-1)) ** (n - i))  # (1, n_pts)
        out.append(term.unsqueeze(-1))
    return torch.cat(out, dim=-1)  # (1, n_pts, n+1)

def curve_distance_loss(pred_curves, gt_curves):
    pred_points = bezier_curve(pred_curves)  # (B, N, 2)
    gt_points = bezier_curve(gt_curves)      # (B, N, 2)
    interp_loss = F.mse_loss(pred_points, gt_points)
    ctrl_loss = F.mse_loss(pred_curves, gt_curves)
    return interp_loss + 0.5 * ctrl_loss

# def curve_distance_loss(pred_curves, gt_curves):
#     # You need to implement this based on control & interpolated point distances
#     return torch.tensor(0.0, requires_grad=True).to(pred_curves.device)

def confidence_score_loss(pred_scores, gt_scores):
    bce = torch.nn.BCEWithLogitsLoss()
    return bce(pred_scores, gt_scores)

def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrinks learning rate by a specified factor.

    :param optimizer: optimizer whose learning rate must be shrunk.
    :param shrink_factor: factor in interval (0, 1) to multiply learning rate with.
    """
    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))


def main(save_path, args):
    # train_file, test_file, val_file = prepare_dataset.get_split(args.data_path)
    train_transform = T.Compose([
        T.ToTensor(),
    ])

    val_transform = T.Compose([
        T.ToTensor(),
    ])  

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
    epochs_since_improvement = 0
    best_val_loss = float('inf')
    
    model = D2GPLand(1024, 1024).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epoch, args.decay_lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.8)

    for epoch in range(args.epoch):
        epoch_running_loss = 0
        epoch_seg_loss = 0

        # training
        model.train()
        # Enable anomaly detection
        torch.autograd.set_detect_anomaly(True)
        # edge_prototypes_model.train()
        for batch_idx, (X_batch, depth, y_batch, gt_curves, gt_scores, induction_map, img_path) in tqdm(enumerate(train_loader)):
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            depth = depth.to(device)
            gt_curves = gt_curves.to(device)
            gt_scores = gt_scores.to(device)
            induction_map = induction_map.to(device)

            output, pred_curves, pred_scores, pred_score_map = model(X_batch, depth)

            y_batch_down = F.interpolate(y_batch, size=(256, 256), mode='nearest').clone()

            seg_loss = dice_loss(output, y_batch_down) + bce_loss(output, y_batch_down)
            # l_ind = proposal_induction_loss(pred_score_map, induction_map)
            # downsample GT induction map to match pred_score_map size
            induction_map_ds = F.interpolate(induction_map, size=pred_score_map.shape[2:], mode='nearest').to(device)
            l_ind = proposal_induction_loss(pred_score_map, induction_map_ds)
            l_crv = curve_distance_loss(pred_curves, gt_curves)
            l_cs = confidence_score_loss(pred_scores, gt_scores.unsqueeze(1))

            epoch_t = max(0, epoch - 10)
            lambda_d = 1 - torch.sigmoid(torch.tensor(epoch_t / 2.0)).item()
            lambda_s, lambda_ind, lambda_crv, lambda_cs = 10.0, 1.0, 1.0, 1.0

            loss = lambda_d * (lambda_s * seg_loss + lambda_ind * l_ind) + \
                (1 - lambda_d) * (lambda_crv * l_crv + lambda_cs * l_cs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_running_loss += loss.item()
            epoch_seg_loss += seg_loss.item()

        # ===================log========================
        print('epoch [{}/{}], loss:{:.4f}'
              .format(epoch, args.epoch, epoch_running_loss / (batch_idx + 1)))
        print('epoch [{}/{}], seg loss:{:.4f}'
              .format(epoch, args.epoch, epoch_seg_loss / (batch_idx + 1)))

        # validation
        model.eval()
        validation_IOU = []
        mDice = []
        val_loss_total = 0


        with torch.no_grad():
            for X_batch, depth, y_batch, _, _, _, _ in tqdm(val_loader):
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                depth = depth.to(device)

                output, _, _, _  = model(X_batch, depth)
                y_batch_down = F.interpolate(y_batch, size=(256, 256), mode='nearest').clone()
                val_loss = dice_loss(output, y_batch_down) + bce_loss(output, y_batch_down)
                val_loss_total += val_loss.item()

                output = torch.argmax(torch.softmax(output, dim=1), dim=1)
                # y_batch = F.interpolate(y_batch, size=(256, 256), mode='nearest').clone()
                y_batch = torch.argmax(y_batch_down, dim=1)

                tmp = output.detach().cpu().numpy()
                tmp2 = y_batch.detach().cpu().numpy()
                tmp = tmp[0]
                tmp2 = tmp2[0]

                pred = np.array([tmp == i for i in range(4)]).astype(np.uint8)
                gt = np.array([tmp2 == i for i in range(4)]).astype(np.uint8)
                # print("pred shape:", pred.shape, "gt shape:", gt.shape)

                iou, dice = evaluation(pred[1:].flatten(), gt[1:].flatten())

                validation_IOU.append(iou)
                mDice.append(dice)

        print(np.mean(validation_IOU))
        print(np.mean(mDice))


        if np.mean(mDice) > best_dice:
            best_dice = np.mean(mDice)
            torch.save(model.state_dict(), save_path + "best_model_path_v15_bcr_dora_b4.pth")
            print("save best model at", save_path + "best_model_path_v15_bcr_dora_b4.pth")
        print("best dice is:{:.4f}".format(best_dice))

        # adjust learning rate based on validation loss
        val_loss = val_loss_total / len(val_loader)
        scheduler.step(val_loss)
        print('Current learning rate:', optimizer.param_groups[0]['lr'])
    # scheduler.step()


if __name__ == '__main__':

    seed_everything(42)

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=3e-5)
    parser.add_argument('--decay_lr', type=float, default=1e-6)
    parser.add_argument('--epoch', type=int, default=150)
    parser.add_argument('--num_landmark', type=int, default=3)
    parser.add_argument('--data_path', type=str, default='../../L3D_Dataset')
    args = parser.parse_args()

    save_path = 'results/'
    os.makedirs(save_path, exist_ok=True)

    main(save_path, args=args)