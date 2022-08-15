import json
import os
import cv2
from torch.utils.data import DataLoader, Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from generate import generate_images
from modules.utils import generate_id
import glob
import logging


class ImageDataset(Dataset):
    def __init__(self, image_paths, labels, dataset_type, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.ds_type = dataset_type
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_filepath = self.image_paths[idx]
        image = cv2.imread(image_filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            image = self.transform(image=image)["image"]
        img_filename = os.path.split(image_filepath)[-1]

        if self.ds_type != 'test':              # Return image and label (PTC=1.0, Non-PTC=0.0)
            img_lbl = [y for x, y in self.labels if x == img_filename][0]
            return image, img_lbl
        else:                                   # Also return patient no. and diagnosis if testing model
            img_lbl = [y for x, y, _, _ in self.labels if x == img_filename][0]
            patient = [z for x, _, z, _ in self.labels if x == img_filename][0]
            diagnosis = [a for x, _, _, a in self.labels if x == img_filename][0]
            return image, img_lbl, patient, diagnosis


def dataloader(image_src, labels, cv_split, dataset_type, img_crop, batch_size,
               workers, gan_params, run_id):

    # Return the filenames of the images in requested train/val/test set:
    dataset_fnames = json.load(open(cv_split))[dataset_type]

    # Defining image paths for FDA transform:
    image_paths = [image_src + lbl for lbl in dataset_fnames]

    if gan_params is not None:
        gan_params = json.load(open(gan_params))
        gan_dir = init_gan_data(run_id, gan_params)                             # Create GAN Images
        gan_paths = glob.glob(gan_dir+"/*")                                     # Return list of all created GAN images
        image_paths = image_paths + gan_paths                                   # Append to real image paths list
        gan_lbls = [int(x.split('_')[-1].split('.')[0]) for x in gan_paths]     # Return image label from path name
        gan_f = [os.path.split(x)[1] for x in gan_paths]                        # Return GAN filenames
        gan_comb = list(map(lambda x, y:[x, y], gan_f, gan_lbls))               # Combine GAN filenames and GAN labels
        labels = labels + gan_comb                                              # Update existing labels list
    else:
        gan_dir = None

    # Load the required transforms and the full image paths:
    trans = define_transforms(dataset_type, image_paths, img_crop)

    # Initialize the dataset:
    img_dataset = ImageDataset(image_paths=image_paths, labels=labels,
                               dataset_type=dataset_type, transform=trans)

    # Create dataloader object:
    dataloader = DataLoader(img_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=workers,
                            )

    return dataloader, gan_dir


def init_gan_data(run_id, gan_params):
    # Create a temporary gan image directory
    gan_dir = generate_id('all_imgs/tmp/', run_id)

    # Get location of saved GAN network from gan_params:
    network_pkl = gan_params['network'][0]

    # Create fake GAN-generated images for each class
    for i, c in enumerate(gan_params['classes']):
        logging.info("Generating GAN images for class: ", c)
        generate_images(network_pkl=network_pkl, seeds=gan_params['seeds'][i],
                        outdir=gan_dir, class_idx=c, bi_class_id=gan_params['class_map'][i])

    return gan_dir


def define_transforms(dataset_type, image_paths, img_crop=512):
    """
    Defines images transforms to be applied to the images.
    -- Dataset_type: train, val or test
    -- image_src: Source dir of image files
    -- dataset_fnames: filenames for dataset images, i.e. '1a.jpeg'
    -- img_crop: Size of the random crop of original image (0 - no cropping)
    """
    # Transformations replicated from Bohland et al. Github code:
    # https://github.com/moboehland/thyroidclassification/blob/main/DLC/classification_module.py

    if dataset_type == 'train':
        # Define the transforms to be performed on the training set:
        trans = A.Compose([
                        A.Flip(p=0.5),
                        A.Rotate(p=0.5),
                        (lambda x: A.RandomCrop(x, x, p=1) if x > 0 else A.RandomCrop(x, x, p=0))(img_crop),
                        A.augmentations.domain_adaptation.FDA(image_paths, beta_limit=0.05, p=0.5),
                        A.OneOf([A.CLAHE(p=0.33),
                                A.RandomBrightnessContrast(p=0.33)],
                                p=0.5),
                        A.Blur(p=0.25),
                        A.GaussNoise(p=0.25),
                        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                        ToTensorV2()
                    ])

    else:
        # And more basic transforms to be performed on the validation and test sets:
        trans = A.Compose([
                    (lambda x: A.RandomCrop(x, x, p=1) if x > 0 else A.RandomCrop(x, x, p=0))(img_crop),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2()
                ])

    return trans


