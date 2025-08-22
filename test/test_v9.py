import argparse
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms as T
from tqdm import tqdm

from utils.metrics import evaluation
from utils import prepare_dataset
from utils.dataset_test import LandmarkDataset#, save_and_show_img
from models.D2GP_v14 import D2GPLand, Edge_Prototypes

import numpy as np

from surface_distance.surface_distance.metrics import compute_surface_distances, compute_average_surface_distance, compute_robust_hausdorff, compute_surface_overlap_at_tolerance, compute_surface_dice_at_tolerance, compute_dice_coefficient


def main(save_path, args):
    # train_file, test_file, val_file = prepare_dataset.get_split(args.data_path)
    val_transform = T.Compose([
        T.ToTensor()
    ])

    # test_dataset = LandmarkDataset(test_file, transform=val_transform, mode='test')
    test_dataset = LandmarkDataset(root=(args.data_path+'/Test'), transform=val_transform, mode='test')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    device = torch.device("cuda")
    print('Training on:', torch.cuda.get_device_name(0), 'test sample size:', len(test_dataset))


    model = D2GPLand(1024, 1024).to(device)

    # model_checkpoint = torch.load(args.model_path)
    # model.load_state_dict(model_checkpoint['model'])
    # edge_prototypes_model = Edge_Prototypes(num_classes=3, feat_dim=256).to(device)
    # prototype_checkpoint = torch.load(args.prototype_path)
    # edge_prototypes_model.load_state_dict(prototype_checkpoint['model'])

    model_checkpoint = torch.load(args.model_path)
    model.load_state_dict(model_checkpoint)
    # edge_prototypes_model = Edge_Prototypes(num_classes=3, feat_dim=256).to(device)
    # prototype_checkpoint = torch.load(args.prototype_path)
    # edge_prototypes_model.load_state_dict(prototype_checkpoint)

    model.eval()
    # edge_prototypes_model.eval()

    validation_IOU = []
    mDice = []
    results = {
            "ASSD": [],
            "Hausdorff": [],
            "SurfaceOverlap": [],
            "SurfaceDice": [],
            "VolumetricDice": []
        }
    tolerance_mm = 1.0
    percent = 95
    spacing_mm = (10, 10) 

    
    for X_batch, depth, y_batch, name in tqdm(test_loader):
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        depth = depth.to(device)
        # prototypes = edge_prototypes_model()

        # output, feature, edge_out = model(X_batch, depth, prototypes)
        output, feature = model(X_batch, depth)
        output = F.interpolate(output, size=(1024, 1024), mode='nearest').clone()
        output = torch.argmax(torch.softmax(output, dim=1), dim=1)
        y_batch = torch.argmax(y_batch, dim=1)

        tmp2 = y_batch.detach().cpu().numpy()
        tmp = output.detach().cpu().numpy()
        tmp = tmp[0]
        tmp2 = tmp2[0]

        pred = np.array([tmp == i for i in range(4)]).astype(np.uint8)
        gt = np.array([tmp2 == i for i in range(4)]).astype(np.uint8)

        iou, dice = evaluation(pred[1:].flatten(), gt[1:].flatten())

        gt_mask =np.array([tmp == i for i in range(4)]).astype(bool)
        pred_mask = np.array([tmp2 == i for i in range(4)]).astype(bool)
        gt_mask =  gt_mask[0]
        pred_mask = pred_mask[0]

        # 檢查有效性
        if not gt_mask.any():
            raise ValueError("Ground truth mask is empty.")
        if not pred_mask.any():
            raise ValueError("Prediction mask is empty.")

        # 檢查形狀匹配
        assert gt_mask.shape == pred_mask.shape, \
            f"Shape mismatch: gt_mask {gt_mask.shape}, pred_mask {pred_mask.shape}"
        assert len(spacing_mm) == gt_mask.ndim, \
            f"Spacing dimensions do not match mask dimensions. Spacing: {spacing_mm}, mask dimensions: {gt_mask.ndim}"

        # 檢查是否有重疊區域
        if not (gt_mask & pred_mask).any():
            print("Warning: No overlap between ground truth and prediction masks.")

        # 計算表面距離
        try:
            surface_distances = compute_surface_distances(gt_mask, pred_mask, spacing_mm)
        except ZeroDivisionError:
            print("Error: Division by zero in distance computation.")
            surface_distances = None

        # 計算表面距離
        surface_distances = compute_surface_distances(gt_mask, pred_mask, spacing_mm)

        # 計算指標
        average_distances = compute_average_surface_distance(surface_distances)
        hausdorff_distance = compute_robust_hausdorff(surface_distances, percent)
        surface_overlap = compute_surface_overlap_at_tolerance(surface_distances, tolerance_mm)
        surface_dice = compute_surface_dice_at_tolerance(surface_distances, tolerance_mm)
        volumetric_dice = compute_dice_coefficient(gt_mask, pred_mask)
        

        # 儲存結果
        results["ASSD"].append(average_distances)
        # results["Hausdorff"].append(hausdorff_distance)
        # results["SurfaceOverlap"].append(surface_overlap)
        # results["SurfaceDice"].append(surface_dice)
        # results["VolumetricDice"].append(volumetric_dice)

        validation_IOU.append(iou)
        mDice.append(dice)
        

        # toprint = save_img(tmp)
        # comparison_img = save_img(tmp, tmp2)

        # # save comparison image
        # filename = os.path.splitext(os.path.basename(name[0]))[0] + '_compare.png'
        # cv2.imwrite(os.path.join(save_path, filename), cv2.cvtColor(comparison_img, cv2.COLOR_RGB2BGR))

        # toprint = save_img(tmp)
        # cv2.imwrite(save_path + '/' + str(name).split('/', 6)[-1].replace('/', '_')[:-2], toprint)

        # Save image
        save_dir = save_path
        filename_base = os.path.splitext(os.path.basename(name[0]))[0]
        img_save_path = os.path.join(save_dir, f"{filename_base}_compare.png")

        # save_and_show_img(
        #     image=np.transpose(X_batch[0].cpu().numpy(), (1, 2, 0)) * 255,
        #     pred_mask=output[0].detach().cpu().numpy(),
        #     gt_mask=y_batch[0].detach().cpu().numpy(),
        #     filename=name[0],
        #     loss=dice.item(),
        #     save_path=img_save_path
        # )
        # print("Saved comparison image:", img_save_path)
    print("Validation IOU:", np.mean(validation_IOU))
    print("Mean Dice:", np.mean(mDice))
   
    print("ASSD:", np.mean(results["ASSD"]))
    # print("Hausdorff:", np.mean(results["Hausdorff"]))
    # print("Surface Overlap:", np.mean(results["SurfaceOverlap"]))
    # print("Surface Dice:", np.mean(results["SurfaceDice"]))
    # print("Volumetric Dice:", np.mean(results["VolumetricDice"]))
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default=None)
    # parser.add_argument('--prototype_path', default=None)
    parser.add_argument('--data_path', type=str, default='../../L3D_Dataset')
    args = parser.parse_args()
    os.makedirs('test_results/', exist_ok=True)

    save_path = 'test_results_DA/'
    os.makedirs(save_path, exist_ok=True)

    main(save_path, args=args)
