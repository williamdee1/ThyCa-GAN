# Thy-GAN 
## Can GAN-Generated Images Help Bridge the Domain Gap between Thyroid Histopathology Image Datasets?

Abstract: *TBC*

## Requirements

## Data repository

| Path | Description
| :--- | :----------
| [Böhland et al. (2021)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8457451/) |  Reference Paper (Based on Tharun Thompson Dataset)
| [Nikiforov Data](http://image.upmc.edu:8080/NikiForov%20EFV%20Study/BoxA/view.apml?listview=1) |  Source of Nikiforov External Dataset Samples
| [TCGA-THCA Data](https://portal.gdc.cancer.gov/projects/TCGA-THCA) |  Source of TCGA External Dataset Samples
| [StyleGAN2-ADA Pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch) | StyleGAN2-ADA Github Repository
| &ensp;&ensp;&boxvr;&nbsp; [styleGAN2-ada.pdf](https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/ada-paper.pdf) | StyleGAN2-ADA Research Paper

## Getting Started

The original images in the Tharun Thompson dataset are 1916x1053 pixels. StyleGAN2-ADA's dataset_tool.py automatically crops 512x512 images from the center of the source dataset. 

If the source data is larger than 512x512, to ensure all data is being used to train the GAN, the images can initially be split using [split_dataset.py](./split_dataset.py) as follows:

```.bash
# Split source data into 512x512 crops:
python split_dataset.py --source data_dir --labels data_labels --dest out_dir --crop_size 512
```
![Splitting image](./images/image_split.PNG)

After zipping the resulting image files and dataset.json labels file, this zip file can be pre-processed by StyleGAN2's [dataset_tool.py](https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/dataset_tool.py). 

```.bash
# Perpare images for StyleGAN2-ADA Generative Adversarial Network Training:
python dataset_tool.py --source data_dir/split_data.zip --dest out_dir --transform center-crop --width 512 --height 512
```

## StyleGAN2-ADA Generative Adversarial Network (GAN)

The StyleGAN2-ADA repository should be downloaded. This work utilizes the [official PyTorch implementation](https://github.com/NVlabs/stylegan2-ada-pytorch).

### Training

The GAN was trained with the following arguments:
  - `--cfg=paper512` to mirror the parameter settings used for the BRECAHAD dataset - a small dataset containing breast cancer histopathology images (see StyleGAN2-ADA paper for more detail). 
  - `--cond==1` ensures the GAN is trained using the labels provided, and so is subsequently able to produce images for a given class.
  - `--mirror==1` includes x-flips of each image in the dataset, effectively doubling the training images.
  - `--kimg-25000` sets the GAN to train based on 25 million real/generated images.

```.bash
# Run StyleGAN2-ADA GAN Training:
python train.py --outdir=outdir --data=dataset_tool_output.zip --gpus=4 --cfg=paper512 --cond=1 --mirror=1 --kimg=25000
```

Shell script files used to run the GAN training on the Queen Mary HPC are included in the [shell_scr](.shell_scr/) directory.
