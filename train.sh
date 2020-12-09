#!/bin/bash
#SBATCH -J Pix2Pix     # Name that will show up in squeue
#SBATCH --gres=gpu:1        # Request 1 GPU "generic resource"
#SBATCH --time=3-22:00       # Max job time is 2 days
#SBATCH --output=%N-%j.out   # Terminal output to file named (hostname)-(jobid).out
#SBATCH --partition=long     # long partition (allows up to 7 days runtime)

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
srun --qos=overcap python train.py --dataroot /project/aksoy-lab/Sepideh/pix2pix/path/to/data --name  cycle_d65_relighttoambient_lr2_mahdi_resnet12_linear_dflash --model cycle_pix2pix_lab --direction AtoB --display_id -1 --dataset_mode alignedlab --lambda_comp 0 --midas 0 --random 0 --netG resnet_12blocks --D_flash 1.0
#srun --qos=overcap python test.py --dataroot /project/aksoy-lab/Sepideh/pix2pix/path/to/data --name cycle_d65_relighttoambient_lr2_mahdi_resnet12_ --model cycle_pix2pix_lab --direction AtoB --dataset_mode alignedlab --midas 0 --random 0 --netG resnet_12blocks  