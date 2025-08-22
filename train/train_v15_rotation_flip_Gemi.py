import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms as T
from tqdm import tqdm
import random
import torch.nn as nn

from utils.metrics import _dice_loss, evaluation, contrastive_loss
from utils import prepare_dataset
from utils.dataset_rotation_flip_BCR import LandmarkDataset
from models.D2GP_v15_CA_v2_BCR import D2GPLand


print ("v15 lora, CA v2, add rotation and flip, loss Gemi, lr: 5e-5") 

def seed_everything(seed=42):   
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def _dice_loss(pred, target, smooth=1.):
    """
    Calculates the Dice loss.
    """
    pred = torch.sigmoid(pred)
    pred_flat = pred.contiguous().view(-1)
    target_flat = target.contiguous().view(-1)
    intersection = (pred_flat * target_flat).sum()
    dice_score = (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
    return 1. - dice_score
 

class ProposalInductionLoss(nn.Module):
    """
    Implementation of the Proposal Induction Loss.
    This loss guides the model to identify the central region of landmarks.
    """
    def __init__(self):
        super(ProposalInductionLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
 
    def create_induction_map(self, target_mask):
        """
        Creates the induction map from the ground truth mask.
        In the paper, this is a map with 1s at GT landmark midpoints.
        Here, we'll approximate this by finding the center of the largest contour.
        """
        induction_map = torch.zeros_like(target_mask)
        for i in range(target_mask.size(0)): # Iterate over batch
            for j in range(target_mask.size(1)): # Iterate over channels/landmarks
                mask_slice = target_mask[i, j].cpu().numpy()
                if mask_slice.sum() > 0:
                    # A simple approximation for the midpoint
                    coords = torch.nonzero(target_mask[i,j])
                    if coords.numel() > 0:
                        mid_point = coords.float().mean(dim=0).int()
                        induction_map[i, j, mid_point[0], mid_point[1]] = 1
        return induction_map.to(target_mask.device)
 
    def forward(self, pred_logits, target_mask):
        induction_target = self.create_induction_map(target_mask)
        return self.bce_loss(pred_logits, induction_target)


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
    # train_dataset = LandmarkDataset(root=(args.data_path+'/Train'), transform=train_transform, mode='train')
    # val_dataset = LandmarkDataset(root=(args.data_path+'/Val/Val'), transform=val_transform, mode='val')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    print('Training on:', torch.cuda.get_device_name(0), 'train sample size:', len(train_dataset), 'val sample size:', len(val_dataset), 'batch:', args.batch_size)

    device = torch.device("cuda")
    # bce_loss = torch.nn.BCEWithLogitsLoss()
    # cl_loss = contrastive_loss()

    # ---------- loss functions ----------
    semantic_segmentation_loss = _dice_loss
    bce_loss = torch.nn.BCEWithLogitsLoss()
    proposal_induction_loss = ProposalInductionLoss().to(device)
 
    # Placeholders for losses that require Bezier curve predictions from the model
    # classification_loss = FocalLoss() # Assumes you have a FocalLoss implementation
    # curve_distance_loss = CurveDistanceLoss() # Assumes you have this implementation
 
    # Loss weights from the paper 
    lambda_s = 10.0
    lambda_ind = 1.0
    lambda_cs = 1.0  # For classification loss
    lambda_crv = 1.0 # For curve distance loss
    # ------------------------------------

    def curve_distance_loss(pred, target):
        return F.mse_loss(pred, target)

    best_dice = 0

    model = D2GPLand(1024, 1024).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epoch, args.decay_lr)

    for epoch in range(args.epoch):
        epoch_running_loss = 0
        epoch_seg_loss = 0
        epoch_induction_loss = 0

        # trainng
        model.train()
        # Enable anomaly detection
        torch.autograd.set_detect_anomaly(True)
        # edge_prototypes_model.train()
        for batch_idx, (X_batch, depth, y_batch, gt_curves, gt_scores, induction_map, path) in tqdm(enumerate(train_loader)):
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            depth = depth.to(device)
            gt_curves = gt_curves.to(device)
            gt_scores = gt_scores.to(device)

            output, pred_curves, pred_scores, _ = model(X_batch, depth)

            y_batch_down = F.interpolate(y_batch, size=(256, 256), mode='nearest').clone()

            # Semantic Segmentation Loss (L_s)
            L_s = semantic_segmentation_loss(output, y_batch_down) + bce_loss(output, y_batch_down)

            # Proposal Induction Loss (L_ind)
            L_ind = proposal_induction_loss(output, y_batch_down)

            # Classification & Curve Losses (L_cs, L_crv)
            L_cs = bce_loss(pred_scores, gt_scores.unsqueeze(1))
            L_crv = curve_distance_loss(pred_curves, gt_curves)

            # Decay coefficient (lambda_d)
            lambda_d = 1.0 - torch.sigmoid(torch.tensor((epoch - 10) / 2.0)).item()

            # Total Loss
            loss = (lambda_d * (lambda_s * L_s + lambda_ind * L_ind)) + \
                ((1 - lambda_d) * (lambda_cs * L_cs + lambda_crv * L_crv))
            # ---------------------------------------------------------------

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # epoch_running_loss += loss.item()
            # epoch_seg_loss += seg_loss.item()
            epoch_running_loss += loss.item()
            epoch_seg_loss += L_s.item()
            epoch_induction_loss += L_ind.item()
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
            for X_batch, depth, y_batch, _, _, _, _ in tqdm(val_loader):
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                depth = depth.to(device)

                output, _, _, _ = model(X_batch, depth)
                output = torch.argmax(torch.softmax(output, dim=1), dim=1)
                y_batch = F.interpolate(y_batch, size=(256, 256), mode='nearest').clone()
                y_batch = torch.argmax(y_batch, dim=1)

                tmp2 = y_batch.detach().cpu().numpy()
                tmp = output.detach().cpu().numpy()
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
            torch.save(model.state_dict(), save_path + "best_model_path_v15_loss_Gemi.pth")
            print("save best model at", save_path + "best_model_path_v15_loss_Gemi.pth")
        print("best dice is:{:.4f}".format(best_dice))
    scheduler.step()


if __name__ == '__main__':

    seed_everything(42)

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=4)
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