# %%

import matplotlib.pyplot as plt
import os
from glob import glob
import numpy as np
import cv2
from PIL import Image
import random

# %%
# Convertit les images en nuance de gris

img = Image.open("datasets/images/FDS.jpg")
BaW = img.convert("L")
BaW.save("datasets/images_BW/FDS_BW.png")


img = Image.open("datasets/images/Augustin_Louis_Cauchy.jpg")
BaW = img.convert("L")
BaW.save("datasets/images_BW/Augustin_Louis_Cauchy_BW.png")


img = Image.open("datasets/images/Pluton.jpg")
BaW = img.convert("L")
BaW.save("datasets/images_BW/Pluton_BW.png")

# %% Gaussian Noise
#  Ajout de bruit gaussien (sigma = 10, 25, 50, 75, 170)


def GaussianNoise(image):
    """
    Return six images with gaussian noise,
    sigma = 10, 25, 50, 75, 170
    """
    noisy_images = []
    sigmas = np.array([10, 25, 50, 75, 170])

    for i in sigmas:
        row, col, ch = image.shape
        mean = 0
        sigma = i
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = image + gauss
        noisy_images.append(noisy)

    return noisy_images


def SavNoise(noisy_images, filename):
    for i in range(len(noisy_images)):
        cv2.imwrite(
            f'datasets/noisy_images/{filename}_AWG_{i}.jpg', noisy_images[i])


img1 = cv2.imread("datasets/images_BW/FDS_BW.png")
img2 = cv2.imread("datasets/images_BW/Augustin_Louis_Cauchy_BW.png")
img3 = cv2.imread("datasets/images_BW/Pluton_BW.png")

AWG_img1 = GaussianNoise(img1)
AWG_img2 = GaussianNoise(img2)
AWG_img3 = GaussianNoise(img3)


SavNoise(AWG_img1, 'FDS')
SavNoise(AWG_img2, 'Augustin_Louis_Cauchy')
SavNoise(AWG_img3, 'Pluton')


# %%

p = 0.05


def sp_noise(image, prob):
    '''
    Add salt and pepper noise to image
    prob: Probability of the noise
    '''
    output = np.zeros(image.shape, np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output


sp_img1 = sp_noise(img1, p)
sp_img2 = sp_noise(img2, p)
sp_img3 = sp_noise(img3, p)


cv2.imwrite('datasets/noisy_images/sp_FDS.jpg', sp_img1)
cv2.imwrite('datasets/noisy_images/sp_Augustin_Louis_Cauchy.jpg', sp_img2)
cv2.imwrite('datasets/noisy_images/sp_Pluton.jpg', sp_img3)
# %%

dir_path = os.path.dirname(os.path.realpath(__file__))

files_path = os.path.join(dir_path, 'datasets/noisy_images')
files = glob(os.path.join(files_path, '*.jpg'))
imagessss = []
for im in files:
    imagessss.append(Image.open(im))


# %%


def plot_gallery(title, images, n_col, n_row, cmap=plt.cm.gray):
    plt.figure(figsize=(2. * n_col, 2.26 * n_row))
    plt.suptitle(title, size=16)
    for i, comp in enumerate(images):
        plt.subplot(n_row, n_col, i + 1)
        vmax = max(comp.max(), -comp.min())
        plt.imshow(comp.reshape(image_shape), cmap=cmap,
                   interpolation='nearest',
                   vmin=-vmax, vmax=vmax)
        plt.xticks(())
        plt.yticks(())
    plt.subplots_adjust(0.01, 0.05, 0.99, 0.93, 0.04, 0.)


plot_gallery(title="dfhdh", images=imagessss, n_col=6, n_row=3)
# %%
_, axes = plt.subplots(nrows=3, ncols=6, figsize=(10, 3))
for ax, image in zip(axes, imagessss):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r)

# %%
_, axs = plt.subplots(1, 6, figsize=(12, 12))
axs = axs.flatten()
type = ["AWG  σ = 10", "AWG  σ = 25", "AWG  σ = 50",
        "AWG  σ = 75", "AWG  σ = 175", 'Salt and Pepper']
i = 0
for img, ax in zip(imagessss[:5] + [imagessss[15]], axs):
    ax.imshow(img)
    ax.axes.set_title(type[i])
    i += 1
    ax.set_axis_off()
plt.show()
# %%

# %%
