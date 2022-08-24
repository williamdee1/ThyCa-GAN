#!/bin/bash
set -e
#$ -cwd
#$ -j y
#$ -pe smp 32                    # 8 cores per GPU
#$ -l h_rt=36:0:0               # Runtime length H:M:S
#$ -l h_vmem=11G                # 11 * 8 = 88G total RAM
#$ -l gpu=4                     # Request 'X' GPUs
#$ -m bea                       # Emails alerts at job start, finish and abortion
#$ -M w.t.dee@se21.qmul.ac.uk   # Email address for alerts

module load python
module load cudnn/8.1.1-cuda11.2
module load gcc/10.2.0

#virtualenv ~/GAN_Project/gan_env

source ~/GAN_Project/gan_env/bin/activate

#pip3 install --upgrade pip

#pip3 install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

#pip3 install -r reqs.txt

python dlc_main.py --src_dir=all_imgs/TharunThompson/ --labels=data/bi_dataset.json --out_dir=logs/ --lrd_epc=10 --lrd_fac=0.5 --es_pat=50 \
--split_file=data/full_data_splits/data_split_4.json --run_id='fd_4_200_biGAN' --batch_size=64 --lr=1e-3 --gan_params=data/full_200_gp.json


deactivate

