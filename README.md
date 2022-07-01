# Thy-GAN 
## Can GAN-Generated Images Help Bridge the Domain Gap between Thyroid Histopathology Image Datasets?

Abstract: *TBC*

## Requirements

## Data repository

| Path | Description
| :--- | :----------
| [Böhland et al. (2021)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8457451/) |  Source of Tharun Thompson GAN-Training Dataset
| [Nikiforov Data](http://image.upmc.edu:8080/NikiForov%20EFV%20Study/BoxA/view.apml?listview=1) |  Source of Nikiforov External Dataset Samples
| [TCGA-THCA Data](https://portal.gdc.cancer.gov/projects/TCGA-THCA) |  Source of TCGA External Dataset Samples
| [StyleGAN2-ADA Pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch) | StyleGAN2-ADA Github Repository
| &ensp;&ensp;&boxvr;&nbsp; [styleGAN2-ada.pdf](https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/ada-paper.pdf) | StyleGAN2-ADA Research Paper

## Getting Started

The original images in the Tharun Thompson dataset are 1916x1053 pixels. StyleGAN2-ADA's dataset_tool.py automatically crops 512x512 images from the center of the source dataset. 

If the source data is larger than 512x512, to ensure all data is being used to train the GAN, the images can initially be split using the split_dataset.py as follows:

```.bash
# Split source data into 512x512 crops:
python split_dataset.py --source data_dir --labels data_labels --dest out_dir --crop_size 512
```
![Splitting image](./images/image_split.png)
