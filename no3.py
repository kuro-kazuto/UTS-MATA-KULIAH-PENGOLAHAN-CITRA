# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 10:28:08 2021

@author: galih
UTS PENGCIT
Soal no 3 (Image Denoising with Anisotropic Diffusion
"""
from medpy.filter.smoothing import anisotropic_diffusion
from skimage.util import random_noise
import matplotlib.pylab as plt
from skimage.color import rgb2gray
from skimage.io import imread
import numpy as np

img = rgb2gray(imread('repo_file/galih.png'))
noisy = random_noise(img, var=0.01)
noisy = np.clip(noisy, 0, 1)
plt.figure(figsize=(15,15))
plt.gray()
plt.subplots_adjust(0,0,1,1,0.05,0.05)
plt.subplot(221), plt.imshow(img), plt.axis('off'), plt.title('Original', size=20)
plt.subplot(222), plt.imshow(noisy), plt.axis('off'), plt.title('Noisy', size=20)
diff_out = anisotropic_diffusion(noisy, niter=20, kappa=20, option=1)
plt.subplot(223), plt.imshow(diff_out), plt.axis('off'), plt.title(r'Anisotropic Diffusion (Perona Malik eq 1, iter=20, $\kappa=20$)', size=18)
diff_out = anisotropic_diffusion(noisy, niter=50, kappa=100, option=2)
plt.subplot(224), plt.imshow(diff_out), plt.axis('off'), plt.title(r'Anisotropic Diffusion (Perona Malik eq 2, iter=50, $\kappa=50$)', size=18)
plt.show()

