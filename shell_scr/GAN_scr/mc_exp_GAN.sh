#!/bin/bash
set -e
#$ -cwd
#$ -j y
#$ -pe smp 32                   # 8 cores per GPU
#$ -l h_rt=240:0:0              # Runtime length H:M:S
#$ -l h_vmem=11G                # 11 * 8 = 88G total RAM
#$ -l gpu=4                     # Request 'X' GPUs
#$ -m bea                       # Emails alerts at job start, finish and abortion
#$ -M w.t.dee@se21.qmul.ac.uk   # Email address for alerts

module load python
module load cudnn/8.1.1-cuda11.2
module load gcc/10.2.0

virtualenv ~/style_GAN/stylegan2-ada-pytorch-main/gan_env

source ~/style_GAN/stylegan2-ada-pytorch-main/gan_env/bin/activate

pip3 install --upgrade pip

pip3 install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

pip3 install -r reqs.txt

python train.py --outdir=gan_output/ --data=tt_mc_exp.zip --gpus=4 --cfg paper512 --cond=1 --mirror=1 --kimg 25000

deactivate
