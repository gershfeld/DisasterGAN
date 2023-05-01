import numpy as np
from PIL import Image
import time
import datetime
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from torch.autograd import Variable
from torchvision.utils import save_image

from dataset import DisasterDataset
from model.cGAN import Discriminator, GeneratorUNet, weights_init_normal

import scipy
import torchvision.models as models


cuda = True if torch.cuda.is_available() else False
img_height, img_width = 1024, 1024
lr = 0.0002
b1 = 0.5
b2 = 0.999
epoch = 0
n_epochs = 200
batch_size = 4
n_cpu = 8
sample_interval = 400
checkpoint_interval = 10
latent_dim = 100

generator = GeneratorUNet()
discriminator = Discriminator()
criterion_GAN = torch.nn.MSELoss()
criterion_pixelwise = torch.nn.L1Loss()

# for resuming training
checkpoint_G = torch.load("saved_models/generator_latest.pth", map_location=torch.device('cpu'))
checkpoint_D = torch.load("saved_models/discriminator_latest.pth", map_location=torch.device('cpu'))

generator.load_state_dict(checkpoint_G['model_state_dict'])
discriminator.load_state_dict(checkpoint_D['model_state_dict'])

generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

optimizer_G.load_state_dict(checkpoint_G['optimizer_state_dict'])
optimizer_D.load_state_dict(checkpoint_D['optimizer_state_dict'])

transforms = transforms.Compose([
    transforms.Resize((img_height, img_width), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

val_dataloader = DataLoader(
    DisasterDataset(r"dataset", transforms=transforms),
    shuffle=True,
)

dataloader = DataLoader(
    DisasterDataset(r"dataset", transforms=transforms),
    batch_size=batch_size,
    shuffle=True,
)

Tensor = torch.FloatTensor


def sample_images(batches_done):
    # TODO: change the dataloader here not to work on batch
    """Saves a generated sample from the validation set"""
    imgs = next(iter(val_dataloader))
    real_A = Variable(imgs[0].type(Tensor))
    real_B = Variable(imgs[1].type(Tensor))
    Variable(LongTensor(np.random.randint(0, 6, batch_size)))
    disaster = Variable(LongTensor(np.random.randint(0, 6, 1)))
    fake_B = generator(real_A, disaster)
    img_sample = torch.cat((real_A.data, fake_B.data, real_B.data), -2)
    save_image((img_sample * 0.5) + 0.5, "samples/%s.png" % batches_done, nrow=5, normalize=True)


lambda_pixel = 100
patch = (1, img_height // 2 ** 4, img_width // 2 ** 4)

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


def main():
    prev_time = time.time()
    for e in range(n_epochs):
        for i, (pre_images, post_images, label) in enumerate(dataloader):

            valid = Variable(FloatTensor(np.ones((pre_images.size(0), *patch))), requires_grad=False)
            fake = Variable(FloatTensor(np.ones((post_images.size(0), *patch))), requires_grad=False)

            pre_images = Variable(pre_images.type(FloatTensor))
            post_images = Variable(post_images.type(FloatTensor))
            label = Variable(label.type(LongTensor))
            optimizer_G.zero_grad()
            gen_labels = Variable(LongTensor(np.random.randint(0, 6, pre_images.shape[0])))
            fake_B = generator(pre_images, gen_labels)
            pred_fake = discriminator(fake_B, pre_images, label)
            loss_GAN = criterion_GAN(pred_fake, valid)
            # Pixel-wise loss
            loss_pixel = criterion_pixelwise(fake_B, post_images)

            # Total loss
            loss_G = loss_GAN + lambda_pixel * loss_pixel

            loss_G.backward()

            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Real loss
            pred_real = discriminator(post_images, pre_images, label)
            loss_real = criterion_GAN(pred_real, valid)

            # Fake loss
            pred_fake = discriminator(fake_B.detach(), pre_images, label)
            loss_fake = criterion_GAN(pred_fake, fake)

            # Total loss
            loss_D = 0.5 * (loss_real + loss_fake)

            loss_D.backward()
            optimizer_D.step()

            batches_done = e * len(dataloader) + i
            batches_left = n_epochs * len(dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()

            # Logging
            print("\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, pixel: %f, adv: %f] ETA: %s"
                  % (
                      e,
                      n_epochs,
                      i,
                      len(dataloader),
                      loss_D.item(),
                      loss_G.item(),
                      loss_pixel.item(),
                      loss_GAN.item(),
                      time_left,
                  ))
            # If at sample interval save image
            if batches_done % sample_interval == 0:
                sample_images(batches_done)

        if checkpoint_interval != -1 and e % checkpoint_interval == 0:
            # Save model checkpoints
            torch.save(generator.state_dict(), "saved_models/generator_%d.pth" % e)
            torch.save(discriminator.state_dict(), "saved_models/discriminator_%d.pth" % e)

## This is for FID
inception_model = models.inception_v3(pretrained=True)
inception_model.eval()
with torch.no_grad():
    imgs = next(iter(val_dataloader))
    real_A = Variable(imgs[0].type(Tensor))
    disaster = Variable(LongTensor(np.random.randint(0, 6, 2)))
    generated_images = generator(real_A, disaster)


def get_feature_statistics(images, model):
    features = []
    for i in range(images.shape[0]):
        img = torch.FloatTensor(images[i].type(Tensor))
        feat = model(img.unsqueeze(0))[0].view(-1).cpu().detach().numpy()
        features.append(feat)
    features = np.array(features)
    mu = np.mean(features, axis=0)
    cov = np.cov(features, rowvar=False)
    return mu, cov


real_images_mu, real_images_cov = get_feature_statistics(imgs[0], inception_model)
generated_images_mu, generated_images_cov = get_feature_statistics(generated_images, inception_model)


def calculate_fid_score(mu1, cov1, mu2, cov2):
    diff = mu1 - mu2
    covsqrt, _ = scipy.linalg.sqrtm(np.dot(cov1, cov2), disp=False)
    if np.iscomplexobj(covsqrt):
        covsqrt = covsqrt.real
    fid_score = diff.dot(diff) + np.trace(cov1 + cov2 - 2 * covsqrt)
    return fid_score


fid_score = calculate_fid_score(real_images_mu, real_images_cov, generated_images_mu, generated_images_cov)
print(fid_score)

if __name__ == "__main__":
    main()
