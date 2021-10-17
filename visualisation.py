# %%

import numpy as np
import cv2
from PIL import Image
import random
from glob import glob
import os
import matplotlib.pyplot as plt

# %%
# Convertit les images en nuance de gris


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
            f'datasets/visualization/noisy_images/{filename}_AWG_{i}.jpg', noisy_images[i])


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


def visualization(path='datasets/images/FDS.jpg', filename='FDS'):

    img = Image.open(path)
    BaW = img.convert("L")
    BaW.save(f"datasets/visualization/images_BW/{filename}_BW.png")

    img1 = cv2.imread(f"datasets/visualization/images_BW/{filename}_BW.png")
    AWG_img1 = GaussianNoise(img1)
    SavNoise(AWG_img1, filename)
    sp_img1 = sp_noise(img1, 0.05)
    cv2.imwrite(
        f'datasets/visualization/noisy_images/sp_{filename}.jpg', sp_img1)

    dir_path = os.path.dirname(os.path.realpath(__file__))

    files_path = os.path.join(dir_path, 'datasets/visualization/noisy_images')
    files = glob(os.path.join(files_path, '*.jpg'))
    imagessss = []
    for im in files:
        imagessss.append(Image.open(im))

    _, axs = plt.subplots(1, 6, figsize=(12, 12))
    axs = axs.flatten()
    type = ["AWG  σ = 10", "AWG  σ = 25", "AWG  σ = 50",
            "AWG  σ = 75", "AWG  σ = 175", 'Salt&Pepper p = 5%']
    i = 0
    for img, ax in zip(imagessss, axs):
        ax.imshow(img)
        ax.axes.set_title(type[i])
        i += 1
        ax.set_axis_off()
    plt.savefig("Cauchy_noise.pdf")
    plt.show()


visualization(path='datasets/images/Augustin_Louis_Cauchy.jpg',
              filename='Augustin_Louis_Cauchy')
# %%
