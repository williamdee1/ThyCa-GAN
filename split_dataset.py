import json
import os
import glob
import tqdm
from tqdm import tqdm
import argparse
from PIL import Image
import numpy as np
cwd = os.getcwd()


def image_splits(image, crop_size):
    """
    Create a list of images cropped from an original image.
    Overlaps the last vertical/horizontal images dependent on image size.
    """
    # Get image width and height from shape:
    img_h, img_w, _ = image.shape

    # Calculate number of height/width crops:
    w_crops = round(img_w / crop_size)
    h_crops = round(img_h / crop_size)

    w_list = []
    h_list = []

    for w in range(w_crops):
        w_start = crop_size * w
        if w_start + crop_size > img_w:
            w_start = img_w - crop_size
        w_list.append(w_start)

    for h in range(h_crops):
        h_start = crop_size * h
        if h_start + crop_size > img_h:
            h_start = img_h - crop_size
        h_list.append(h_start)

    # Combine the crop locations into one list of splits:
    split_list = []

    for i in h_list:
        for j in w_list:
            split = image[i:i + crop_size, j:j + crop_size]
            split_list.append(split)

    return split_list


def split_dataset():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', help='Directory name for input dataset', required=True)
    parser.add_argument('--labels', help='File name for labels json file', required=True)
    parser.add_argument('--dest', help='Output directory for cropped dataset', required=True)
    parser.add_argument('--crop_size', help='Width x Height dimensions of outputted crops', default=512, type=int)
    opt = parser.parse_args()

    # Get paths of all files in source directory:
    file_paths = glob.glob(os.path.join(opt.source, '*'))

    # Print out some useful information:
    og_img_shape = np.array(Image.open(file_paths[0])).shape
    print("Converting original images of shape %s, into multiple images of shape (%s, %s, 3)." % (og_img_shape,
                                                                                                  opt.crop_size,
                                                                                                  opt.crop_size))

    # Load json file with dataset labels:
    lbl_f = open(opt.labels)
    dset_lbls = json.load(lbl_f)

    # Empty list for new filenames/ labels:
    new_lbls = []

    # Iterate through image files in directory:
    for img_file in tqdm(file_paths, desc='Processing Image Files'):

        # Log filename:
        file_n = os.path.basename(img_file)
        file_id = os.path.splitext(file_n)[0]

        # Return image classification label:
        img_lbl = [x[1] for x in dset_lbls['labels'] if x[0] == file_n][0]

        # Load image as array:
        img = np.array(Image.open(img_file))

        # Return a list of split images:
        img_spl = image_splits(img, opt.crop_size)

        # Save each image split as separate file:
        for i in range(len(img_spl)):
            img = img_spl[i]
            img = Image.fromarray(img)
            save_file = '%s_%s.jpeg' % ( file_id, i)
            save_dest = '%s/%s' % (opt.dest, save_file)
            img.save(save_dest)
            new_lbls.append([save_file, img_lbl])

    # Store new json dataset labels file (in format for dataset_tool.py):
    lab_dict = {}
    lab_dict['labels'] = new_lbls
    json_file_loc = '%s/dataset.json' % opt.dest
    with open(json_file_loc, 'w') as f:
        json.dump(lab_dict, f)


# ----------------------------------------------------------------------------

if __name__ == "__main__":
    split_dataset()
