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
from utils import prepare_dataset
from utils.dataset_flip import LandmarkDataset
from models.D2GP_v14 import D2GPLand, Edge_Prototypes


print ("v_14, add flip") 

def seed_everything(seed=42):   
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

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

    model = D2GPLand(1024, 1024).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epoch, args.decay_lr)

    for epoch in range(args.epoch):
        epoch_running_loss = 0
        epoch_seg_loss = 0
        # epoch_contrastive_loss = 0
        # epoch_edge_loss = 0

        # trainng
        model.train()
        # Enable anomaly detection
        torch.autograd.set_detect_anomaly(True)
        # edge_prototypes_model.train()
        for batch_idx, (X_batch, depth, y_batch, *rest) in tqdm(enumerate(train_loader)):
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            depth = depth.to(device)

            output, _ = model(X_batch, depth)

            y_batch_down = F.interpolate(y_batch, size=(256, 256), mode='nearest').clone()
            seg_loss = _dice_loss(output, y_batch_down) + bce_loss(output, y_batch_down)
            loss = seg_loss

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
            torch.save(model.state_dict(), save_path + "best_model_path_v14_flip.pth")
            print("save best model at", save_path + "best_model_path_v14_flip.pth")
        print("best dice is:{:.4f}".format(best_dice))
    scheduler.step()


if __name__ == '__main__':

    seed_everything(42)

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=3e-5)
    parser.add_argument('--decay_lr', type=float, default=1e-6)
    parser.add_argument('--epoch', type=int, default=60)
    parser.add_argument('--num_landmark', type=int, default=3)
    parser.add_argument('--data_path', type=str, default='../../L3D_Dataset')
    

    args = parser.parse_args()

    save_path = 'results/'
    os.makedirs(save_path, exist_ok=True)

    main(save_path, args=args)