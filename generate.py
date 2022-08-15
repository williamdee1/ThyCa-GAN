# geneate.py is an adapated version of the generate.py module within the StyleGAN2-ADA repository:
# https://github.com/NVlabs/stylegan2-ada-pytorch
# It has been adpated for use within Thy-GAN, but relies on the dnnlib and legacy packages from the original repo.

"""Generate images using pretrained network pickle."""

import os
import re
from typing import List
import dnnlib
import numpy as np
import PIL.Image
import torch
import legacy
import logging

# ----------------------------------------------------------------------------


def num_range(seeds):
    """
    Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.
    """
    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(seeds)
    if m:
        return list(range(int(m.group(1)), int(m.group(2))))
    vals = seeds.split(',')
    return [int(x) for x in vals]

# ----------------------------------------------------------------------------


def generate_images(network_pkl, seeds, outdir, class_idx, bi_class_id,
                    truncation_psi=1, noise_mode='const'):
    """
    Generate images using pretrained network pickle.
    """

    logging.info('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    os.makedirs(outdir, exist_ok=True)

    if seeds is None:
        logging.info('--seeds option is required when not using --projected-w')
    else:
        seeds = num_range(seeds)

    # Labels.
    label = torch.zeros([1, G.c_dim], device=device)
    if G.c_dim != 0:
        if class_idx is None:
            logging.info('Must specify class label with --class when using a conditional network')
        label[:, class_idx] = 1
    else:
        if class_idx is not None:
            logging.info ('warn: --class=lbl ignored when running on an unconditional network')

    # Generate images.
    logging.info('Generating images from GAN seeds...')
    for seed_idx, seed in enumerate(seeds):
        #logging.info('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
        img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(
            f'{outdir}/seed{seed:04d}_{class_idx:d}_{bi_class_id:d}.jpeg')

