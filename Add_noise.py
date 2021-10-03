# %%

import numpy as np
import cv2
from PIL import Image
import random

# %%
img = Image.open("datasets/images/FDS.jpg")
BaW = img.convert("L")
BaW.save("datasets/images_BW/FDS_BW.png")


img = Image.open("datasets/images/Pikachu_surfeur.png")
BaW = img.convert("L")
BaW.save("datasets/images_BW/Pikachu_surfeur_BW.png")


img = Image.open("datasets/images/Augustin_Louis_Cauchy.jpg")
BaW = img.convert("L")
BaW.save("datasets/images_BW/Augustin_Louis_Cauchy_BW.png")


img = Image.open("datasets/images/Pluton.jpg")
BaW = img.convert("L")
BaW.save("datasets/images_BW/Pluton_BW.png")

# %% Gaussian Noise


def GaussianNoise(image):
    """
    Return six images with gaussian noise,
    sigma = 10, 25, 50, 75, 170
    """
    noisy_images = []
    sigmas = np.array([10, 25, 50, 75, 170])

    for i in sigmas:
        row,col,ch= image.shape
        mean = 0
        sigma = i
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        noisy = image + gauss
        noisy_images.append(noisy)

    return noisy_images


def SavNoise(noisy_images, filename):
    for i in range (len(noisy_images)):
        cv2.imwrite(f'datasets/noisy_images/{filename}_AWG_{i}.jpg', noisy_images[i])


img1 = cv2.imread("datasets/images_BW/Pikachu_surfeur_BW.png")
img2 = cv2.imread("datasets/images_BW/FDS_BW.png")
img3 = cv2.imread("datasets/images_BW/Augustin_Louis_Cauchy_BW.png")
img4 = cv2.imread("datasets/images_BW/Pluton_BW.png")

AWG_img1 = GaussianNoise(img1)
AWG_img2 = GaussianNoise(img2)
AWG_img3 = GaussianNoise(img3)
AWG_img4 = GaussianNoise(img4)

SavNoise(AWG_img1, 'Pikachu_surfeur')
SavNoise(AWG_img2, 'FDS')
SavNoise(AWG_img3, 'Augustin_Louis_Cauchy')
SavNoise(AWG_img4, 'Pluton')


# %%

p = 0.05
def sp_noise(image,prob):
    '''
    Add salt and pepper noise to image
    prob: Probability of the noise
    '''
    output = np.zeros(image.shape,np.uint8)
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

sp_img1 = sp_noise(img1,p)
sp_img2 = sp_noise(img2,p)
sp_img3 = sp_noise(img3,p)
sp_img4 = sp_noise(img4,p)

cv2.imwrite('datasets/noisy_images/sp_Pikachu_sufeur.jpg', sp_img1)
cv2.imwrite('datasets/noisy_images/sp_FDS.jpg', sp_img2)
cv2.imwrite('datasets/noisy_images/sp_Augustin_Louis_Cauchy.jpg', sp_img3)
cv2.imwrite('datasets/noisy_images/sp_Pluton.jpg', sp_img4)
# %%
