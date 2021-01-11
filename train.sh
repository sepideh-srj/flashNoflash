#!/bin/bash
#SBATCH -J FlashNoFlash     # Name that will show up in squeue
#SBATCH --gres=gpu:1        # Request 1 GPU "generic resource"
#SBATCH --time=6-22:00       # Max job time is 7 days
#SBATCH --output=%N-%j.out   # Terminal output to file named (hostname)-(jobid).out
#SBATCH --partition=long     # long partition (allows up to 7 days runtime)
#SBATCH -w cs-venus-03
#SBATCH --qos=overcap
# The SBATCH directives above set options similarly to command line arguments to srun
# Run this script with: sbatch my_experiment.sh
# The job will appear on squeue and output will be written to the current directory
# You can do tail -f <output_filename> to track the job.
# You can kill the job using scancel <job_id> where you can find the <job_id> from squeue

# Your experiment setup logic here
source /home/ssarajia/miniconda3/etc/profile.d/conda.sh
conda activate pix2pix
hostname
echo ------------Starting Training---------
echo $CUDA_AVAILABLE_DEVICES
# sotoon please run these five cases:
#srun --qos=overcap python train.py --dataroot /project/aksoy-lab/Sepideh/data_big --name  final_all --model cycle_pix2pix_lab --direction AtoB --display_id -1 --dataset_mode greatrandom --midas 1 --midas_normal 0 --midas_flash 0 --lambda_comp 1000 --random 0 --netG resnet_12blocks --D_flash 1 --lambda_color_uv 1000
#srun --qos=overcap python train.py --dataroot /project/aksoy-lab/Sepideh/data_big --name  final_withoutcomp --model cycle_pix2pix_lab --direction AtoB --display_id -1 --dataset_mode greatrandom --midas 1 --midas_normal 0 --midas_flash 0 --lambda_comp 1000 --random 0 --netG resnet_12blocks --D_flash 1 --lambda_color_uv 1000
#srun --qos=overcap python train.py --dataroot /project/aksoy-lab/Sepideh/data_big --name  final_withoutcolorloss --model cycle_pix2pix_lab --direction AtoB --display_id -1 --dataset_mode greatrandom --midas 1 --midas_normal 0 --midas_flash 0 --lambda_comp 1000 --random 0 --netG resnet_12blocks --D_flash 1 --lambda_color_uv 0
#srun --qos=overcap python train.py --dataroot /project/aksoy-lab/Sepideh/data_big --name  final_withoutcolorlossandcomp_256 --model cycle_pix2pix_lab --direction AtoB --display_id -1 --dataset_mode greatrandom --midas 1 --midas_normal 0 --midas_flash 0 --lambda_comp 0 --random 0 --netG resnet_12blocks --D_flash 1 --lambda_color_uv 0 --crop_size 256 --load_size 256 
#srun --qos=overcap python train.py --dataroot /project/aksoy-lab/Sepideh/data_big --name  final_withoutDflash_256 --model cycle_pix2pix_lab --direction AtoB --display_id -1 --dataset_mode greatrandom --midas 1 --midas_normal 0 --midas_flash 0 --lambda_comp 1000 --random 0 --netG resnet_12blocks --D_flash 0 --lambda_color_uv 1000 --crop_size 256 --load_size 256 

srun --qos=overcap python test.py --dataroot /project/aksoy-lab/Sepideh/data_big --name  final_all --model cycle_pix2pix_lab --direction AtoB --dataset_mode greatrandom --midas 1 --midas_normal 0 --midas_flash 0 --lambda_comp 1000 --random 0 --netG resnet_12blocks --D_flash 1 --lambda_color_uv 1000

#srun --qos=overcap python test.py --dataroot /project/aksoy-lab/Sepideh/data --name cycle_toambient_lr2_mahdi_resnet12_linear_newData --model cycle_pix2pix_lab --direction AtoB --dataset_mode alignedlab --midas 0 --midas_normal 0 --random 0  --lambda_comp 0 --netG resnet_12blocks --D_flash 0