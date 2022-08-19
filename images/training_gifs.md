# GIFS of Training


## Binary StyleGAN2-ADA - Training Images

Batches of nine GAN-generated images produced from the same latent vector inputs recorded at intervals throughout GAN training with binary labels. FID score of 5.10 after 19.55 million training images.

<img src="gifs/biGAN.gif" width="500" height="536" class="center"/>


## DC-GAN - Training Failures

Batches of 16 images produced by the DC-GAN throughout training where there is evidence of three different failure scenarios – mode collapse, limited discriminator feedback and limited generator capacity. 

<p float="left">
  <img src="gifs/mode_collapse.gif" width="300" height="300" />
  <img src="gifs/lim_d.gif" width="300" height="300" />
  <img src="gifs/gen_c.gif" width="300" height="300" />
</p>

### Details of Failure Types:
- Mode collapse: where the generator learns to collapse many or the whole latent distribution, z, to a narrow subsection or one value of the real data distribution, x. This results in the GAN-generated images all looking alike.
- Limited Discriminator feedback: the discriminator learns too quickly, often overfitting on the underlying data, and provides very little feedback to the generator for gradient weight updates. The generator fails to approximate any of the underlying distribution which then causes mode collapse, generating objects of random shape.
- Generator lacks capacity: The discriminator provides adequate feedback early in training, but the generator model is not sufficiently complex to construct the mapping from z to x. In this scenario the GAN never manages to produce high fidelity images. 
