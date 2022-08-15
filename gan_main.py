import torch
import torch.nn as nn
import os, time, cv2, json
from datetime import datetime
import time
import numpy as np
import pandas as pd
from statistics import mean
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
from pytorch_fid.fid_score import calculate_fid_given_paths
import matplotlib.pyplot as plt
from modules.basic_G import Generator
from modules.basic_D import Discriminator
from modules.gan_funcs import gan_dataloader
from modules.utils import weights_init_normal
from modules.unet_G import GeneratorUNet

def main():
    # ------------
    #  Parameters
    # ------------
    run_id = '5hr_G1e4_D4e4_GDO_2e1'
    image_dir = os.path.join("images/original/all_imgs/")
    crop_size = 512
    channels = 3
    latent_dim = 100
    opt_b1 = 0.0
    opt_b2 = 0.999
    g_lr = 1e-4
    d_lr = 4e-4
    dropout_G = 0.2	
    batch_size = 64
    workers = 8
    # Training length in no. Images "seen" by models:
    train_len = 1000000
    # Sample interval in no. images (ideally multiple of batch size):
    samp_int = 20000
    # Check for over/under-fitting interval
    minibatch_check = 4
    # Target D(x) output:
    dx_target = 0.6
    # Setting the rate at which to apply augmentations initially and amount to increment when changed:
    p_apply = 0.0
    p_inc = 5e-2
    # Setting minimum FID score to inifite initially:
    min_fid_sc = np.Inf

    # ----------
    #  Initialization
    # ----------
    # Initialize generator and discriminator
    generator = GeneratorUNet(crop_size, latent_dim, dropout=dropout_G)
    discriminator = Discriminator(crop_size, channels)

    # Loss function
    adversarial_loss = torch.nn.BCELoss()

    # Assign to GPU if present:
    gpu_cuda = True if torch.cuda.is_available() else False
    # if cuda:
    #     generator.cuda()
    #     discriminator.cuda()
    #     adversarial_loss.cuda()

    # Assigning model to GPU:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        para_G = nn.DataParallel(generator, device_ids = [0,1,2,3])
        para_D = nn.DataParallel(discriminator, device_ids = [0,1,2,3])

    para_G.to(0)
    para_D.to(0)
    adversarial_loss.to(0)

    # Initialize weights
    # para_G.apply(weights_init_normal)
    # para_D.apply(weights_init_normal)

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=g_lr, betas=(opt_b1, opt_b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=d_lr, betas=(opt_b1, opt_b2))

    Tensor = torch.cuda.FloatTensor if gpu_cuda else torch.FloatTensor

    # ----------
    #  Training
    # ----------
    # Create directories for results:
    date_stamp = datetime.now().strftime("Dt_%d_%m_%H%M")
    gen_img_dir = "images/generated/%s" % date_stamp
    real_imgs_dir = "images/fid_batches/real_imgs/%s" % date_stamp
    fake_imgs_dir = "images/fid_batches/fake_imgs/%s" % date_stamp
    model_dir = "models/%s" % date_stamp
    csv_log_dir = "logs/%s" % date_stamp
    os.makedirs(gen_img_dir, exist_ok=True)
    os.makedirs(real_imgs_dir, exist_ok=True)
    os.makedirs(fake_imgs_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(csv_log_dir, exist_ok=True)

    # Setting a counter to record how many images "seen" during training
    img_counter = 0
    # And the number "seen" within batches
    batch_imgs = 0

    # ----------------------------------------
    #  Hold Results per Image Sample Interval
    # ----------------------------------------
    d_loss_epc = []
    d_x_epc = []
    g_loss_epc = []
    d_g_z_epc = []
    img_batches = []
    p_app_track = []
    fid_scores = []

    while img_counter < train_len:

        # Create dataloader object:
        dataloader = gan_dataloader('pix_geom', p_apply, image_dir, batch_size, workers)
        #print("Dataloader reset at %s, p_apply = %s" % (img_counter, p_apply))

        # ---------------------
        #  Hold Batch Results
        # ---------------------
        if (batch_imgs == 0) or (batch_imgs % samp_int == 0) or (batch_imgs > samp_int):
            # Reset batch counter and stats:
            batch_imgs = 0
            d_loss_btc = []
            d_x_btc = []
            g_loss_btc = []
            d_g_z_btc = []
            # Time length to process "samp_int" images:
            batch_start = time.process_time()

        for i, (imgs, _) in enumerate(dataloader):

            # Adversarial ground truths
            valid = Variable(Tensor(imgs.shape[0], 1).fill_(0.9), requires_grad=False)
            fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)

            # Configure input
            real_imgs = Variable(imgs.type(Tensor)).to(0)

            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()

            # Sample noise as generator input
            z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], latent_dim)))).to(0)

            # Generate a batch of images
            gen_imgs = para_G(z)

            # Loss measures generator's ability to fool the discriminator
            g_loss = adversarial_loss(para_D(gen_imgs), valid)

            # Calculate gradients and update Generator
            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            real_output = para_D(real_imgs)
            fake_output = para_D(gen_imgs.detach())

            # Calculate average D prediction:
            d_x = real_output.mean().item()
            d_g_z = fake_output.mean().item()

            # Calculate losses for D and G:
            real_loss = adversarial_loss(real_output, valid)
            fake_loss = adversarial_loss(fake_output, fake)
            d_loss = (real_loss + fake_loss) / 2

            # Calculate gradients and update Discriminator:
            d_loss.backward()
            optimizer_D.step()

            # ---------------------
            #  Store Batch Results
            # ---------------------
            d_loss_btc.append(d_loss.item())
            d_x_btc.append(d_x)
            g_loss_btc.append(g_loss.item())
            d_g_z_btc.append(d_g_z)

            # Increment image counters:
            img_counter += batch_size
            batch_imgs += batch_size

            # ------------------------------------
            #  Adaptive Discriminator Augmentation
            # ------------------------------------
            # Check for over/under fitting of discriminator every X batches (alter p_apply if over/under fitting):
            if (i % minibatch_check == 0) and (i != 0):
                # If D is over-fitting, D(x) will be > dx_target:
                # Use the average D(x) over the last X batches:
                av_dx = mean(d_x_btc[-minibatch_check:])
                # print("Av d(x) = ", av_dx)
                if av_dx > dx_target:
                    if p_apply == 0.8:
                        pass
                    else:
                        p_apply = min((p_apply + p_inc), 0.8)
                        # Break out of current batch loop to reset dataloader:
                        break
                elif (av_dx < dx_target) and (p_apply > 0.0):
                    p_apply = max((p_apply - p_inc), 0.0)
                    break
                else:
                    pass

            if batch_imgs >= samp_int:
                break

        # ---------------------
        #  Display Results
        # ---------------------

        if (batch_imgs % samp_int == 0) or (batch_imgs > samp_int):
            # Average results across batches and store:
            d_loss_mean = mean(d_loss_btc)
            d_x_mean = mean(d_x_btc)
            g_loss_mean = mean(g_loss_btc)
            d_g_z_mean = mean(d_g_z_btc)

            # Append overall results lists:
            d_loss_epc.append(d_loss_mean)
            d_x_epc.append(d_x_mean)
            g_loss_epc.append(g_loss_mean)
            d_g_z_epc.append(d_g_z_mean)
            img_batches.append(img_counter)
            p_app_track.append(p_apply)

            # ---------------------
            #  Calculate FID
            # ---------------------
            # Saving a batch of real and generated images:
            for r, img in enumerate(imgs):
                save_image(img, '%s/%s.png' % (real_imgs_dir, r), normalize=True)
            for g, g_img in enumerate(gen_imgs):
                save_image(g_img, '%s/%s.png' % (fake_imgs_dir, g), normalize=True)

            fid_sc = calculate_fid_given_paths((real_imgs_dir, fake_imgs_dir),
                                               device='cpu', batch_size=batch_size,
                                               num_workers=workers, dims=2048)

            fid_scores.append(fid_sc)

            time_taken = time.process_time() - batch_start

            # Print log
            print(
                "[Training Images %d/%d] [D loss: %f] [D(x): %f] [G loss: %f] [D(G(z)): %f] [Time Taken: %f]  [P Apply: %f]  [FID: %f]"
                % (img_counter, train_len, mean(d_loss_btc), mean(d_x_btc),
                   mean(g_loss_btc), mean(d_g_z_btc), time_taken, p_apply, fid_sc)
            )

            # --------------------------------
            #  Save Image Batches at Intervals
            # --------------------------------
            print("#---------------> SAVING IMAGES <---------------#")
            time_stamp = datetime.now().strftime("_Time_%H.%M")
            save_image(gen_imgs.data[:batch_size],
                       "%s/Imgs_%d_FID_%d_%s.png" % (gen_img_dir, img_counter, fid_sc, time_stamp),
                       nrow=8, normalize=True)

            # --------------------------------
            #  Save Best Generator Model
            # --------------------------------
            # Saving best model (batch with lowest FID score) throughout training:
            if fid_sc < min_fid_sc:
                # Updating min_val_loss to new lowest:
                min_fid_sc = fid_sc
                # Saving the Generator that produced that FID:
                torch.save(para_G.state_dict(), "%s/%s%s.pth" % (model_dir, run_id, '_Gen_FID'))

            # ---------------------
            #  Storing Overall Metrics
            # ---------------------

            res_dict = {'train_imgs': img_batches, 'd_loss': d_loss_epc, 'd_x': d_x_epc,
                        'g_loss': g_loss_epc, 'd_g_z': d_g_z_epc, 'p_app': p_app_track,
                        'fid': fid_scores}

            df = pd.DataFrame(res_dict)
            log_savefile = "%s/%s.csv" % (csv_log_dir, run_id)
            df.to_csv(log_savefile, index=None)

    # Save generator and discriminator weights at end of training:
    # torch.save(para_G.state_dict(), "%s/%s%s.pth" % (model_dir, run_id, '_Gen'))
    # torch.save(para_D.state_dict(), "%s/%s%s.pth" % (model_dir, run_id, '_Disc'))


if __name__ == "__main__":
    main()
