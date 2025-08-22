#! /usr/bin/bash

# Set memory and GPU requirements for the job
#$ -l tmem=20G
#$  -l gpu=true
#$  -pe gpu 1
#$  -l gpu_type=a6000
#  -l gpu_type=a100_80

# Set the job name and the loggin directories/files
#$ -N test
#$ -o /SAN/medic/SAM2/D2GP/D2GPLand/logs
#$ -e /SAN/medic/SAM2/D2GP/D2GPLand/logs
#$ -wd /SAN/medic/SAM2/D2GP/D2GPLand

# Activate Conda environment
eval "$(/SAN/medic/CARES/mobarak/venvs/anaconda3/bin/conda shell.bash hook)"
conda activate sam2

# conda activate sam_yun
export HF_HOME=/SAN/medic/SAM2/D2GP/D2GPLand/hf_cache


########################################################
## CUDA Environment Setup
########################################################
 
# Add CUDA binary directories to PATH - enables system to find and execute CUDA tools (nvidia-smi, nvcc, etc.)
export PATH=/share/apps/cuda-11.8/bin:/usr/local/cuda-11.8/bin:${PATH}
 
# Set runtime library path - tells system where to find CUDA shared libraries during program execution
# This includes both shared (/share/apps) and local (/usr/local) CUDA installations
export LD_LIBRARY_PATH=/share/apps/cuda-11.8/lib64:/usr/local/cuda-11.8/lib:/lib64:${LD_LIBRARY_PATH}
 
# Set CUDA include directory - specifies location of CUDA header files
# Used during compilation of CUDA programs (if needed)
export CUDA_INC_DIR=/share/apps/cuda-11.8/include
 
# Set compile-time library path - tells compiler where to find libraries during linking
# Similar to LD_LIBRARY_PATH but used at build time instead of runtime
export LIBRARY_PATH=/share/apps/cuda-11.8/lib64:/usr/local/cuda-11.8/lib:/lib64:${LIBRARY_PATH}


# Navigate to the directory containing the scripts
cd /SAN/medic/SAM2/D2GP/D2GPLand


# this is the original test weights script
# python test.py --model_path 'results/model.pt' \
#   --prototype_path 'results/prototype.pt' \
#   --data_path '../../L3D_Dataset'

# python test.py --model_path 'results/best_model_path_D2GP_without_aug.pth' \
#   --prototype_path 'results/best_prototype_path_D2GP_without_aug.pth' \
#   --data_path '../../L3D_Dataset'


# python test_attention.py --model_path 'results/best_model_path_d2gp_again.pth' \
#   --prototype_path 'results/best_prototype_path_d2gp_again.pth' \
#   --data_path '../../L3D_Dataset'

# python test_v15.py --model_path 'results/best_model_path_v15_galore_v4.pth' \
#   --data_path '../../L3D_Dataset'
python test_v15_image.py --model_path 'results/best_model_path_v15_galore_v4.pth' \
  --data_path '../../Lap_Images'

# python test_v18.py --model_path 'results/best_model_path_v18_r256.pth' \
#   --data_path '../../L3D_Dataset'


# python test_v15_BCR.py --model_path 'results/best_model_path_v15_dora_lr_decay.pth' \
#   --data_path '../../L3D_Dataset'



# python train_v15_flip.py
# python train_v15_rotation_flip_Gemi.py
# python train_v15_rotation_flip_galore_v4.py
# python train_v16.py
# python train_v15_rotation_flip_veLora.py
# python train_v15_bcr_dora.py
# python train_v15_rotation_flip_dora.py
# python train_v15_rotation_flip_DA1.py
# python train_v15_rotation_flip_SAM1.py
# python test.py
# python train_v16.py
# python train_v18.py
