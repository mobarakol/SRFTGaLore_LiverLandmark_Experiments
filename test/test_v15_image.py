import argparse
import os
import cv2
import torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms as T
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.D2GP_v15_galore_v4 import D2GPLand
from utils.dataset_test_image import LandmarkDataset, save_and_show_img  # 或自行寫簡化版本的可視化儲存函式



def main(args):
    val_transform = T.Compose([
        T.Resize((1024, 1024)),
        T.ToTensor()
    ])

    test_dataset = LandmarkDataset(root=(args.data_path+'/Val'), transform=val_transform, mode='test')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    device = torch.device("cuda")
    print('Training on:', torch.cuda.get_device_name(0), 'test sample size:', len(test_dataset))

    model = D2GPLand(1024, 1024).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    with torch.no_grad():
        for X_batch, depth, y_batch, name in tqdm(test_loader):
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            depth = depth.to(device)

            output, feature = model(X_batch, depth)
            output = F.interpolate(output, size=(1024, 1024), mode='nearest').clone()
            output = torch.argmax(torch.softmax(output, dim=1), dim=1)
            y_batch = torch.argmax(y_batch, dim=1)


            # Save image
            save_dir = args.save_dir
            filename = os.path.splitext(os.path.basename(name[0]))[0]
            img_save_path = os.path.join(save_dir, f"{filename}_segmentation.png")
            os.makedirs(save_dir, exist_ok=True)

            save_and_show_img(
                    image=np.transpose(X_batch[0].cpu().numpy(), (1, 2, 0)) * 255,
                    pred_mask=output[0].detach().cpu().numpy(),
                    gt_mask=None,  # 如果沒有 ground truth mask，可以設為 None
                    filename=name[0],
                    loss=None,
                    save_path=img_save_path
                )
            print(f"Saved segmentation result to {img_save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='results/best_model_path_v15_galore_v4.pth')
    parser.add_argument('--data_path', type=str, default='../../Lap_Images')
    parser.add_argument('--save_dir', type=str, default='test_results_lap/')
    args = parser.parse_args()
    os.makedirs('test_results_lap/', exist_ok=True)

    save_path = 'test_results_lap/'

    os.makedirs(save_path, exist_ok=True)

    main(args=args)