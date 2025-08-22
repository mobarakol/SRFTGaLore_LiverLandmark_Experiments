import argparse
import os
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T
from tqdm import tqdm

from utils.metrics import evaluation
from utils import prepare_dataset
from utils.dataset_test import LandmarkDataset, save_and_show_img
from models.D2GP_v15 import D2GPLand, Edge_Prototypes


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
    edge_prototypes_model = Edge_Prototypes(num_classes=3, feat_dim=256).to(device)
    prototype_checkpoint = torch.load(args.prototype_path)
    edge_prototypes_model.load_state_dict(prototype_checkpoint)

    model.eval()
    edge_prototypes_model.eval()

    validation_IOU = []
    mDice = []

    for X_batch, depth, y_batch, name in tqdm(test_loader):
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        depth = depth.to(device)
        prototypes = edge_prototypes_model()

        output, feature, edge_out = model(X_batch, depth, prototypes)
        output = torch.argmax(torch.softmax(output, dim=1), dim=1)
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
        # toprint = save_img(tmp)
        comparison_img = save_img(tmp, tmp2)

        # save comparison image
        filename = os.path.splitext(os.path.basename(name[0]))[0] + '_compare.png'
        cv2.imwrite(os.path.join(save_path, filename), cv2.cvtColor(comparison_img, cv2.COLOR_RGB2BGR))

        # toprint = save_img(tmp)
        # cv2.imwrite(save_path + '/' + str(name).split('/', 6)[-1].replace('/', '_')[:-2], toprint)

    print("Validation IOU:", np.mean(validation_IOU))
    print("Mean Dice:", np.mean(mDice))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default=None)
    parser.add_argument('--prototype_path', default=None)
    parser.add_argument('--data_path', type=str, default='../../L3D_Dataset')
    args = parser.parse_args()
    os.makedirs('test_results/', exist_ok=True)

    save_path = 'test_results/'
    os.makedirs(save_path, exist_ok=True)

    main(save_path, args=args)
