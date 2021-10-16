# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 11:22:34 2021

@author: galih
UTS PENGCIT
Soal no 5 (Weiner Filter)
"""

from skimage import color, data, restoration
import scipy.fftpack as fp
from skimage.measure import compare_psnr
from skimage.io import imread
import numpy as np
import matplotlib.pylab as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter

def convolve2d(im, psf, k):
    M, N = im.shape
    freq = fp.fft2(im)
    psf = np.pad(psf, (((M-k)//2,(M-k)//2+1), ((N-k)//2,(N-k)//2+1)), mode='constant') # assumption: min(M,N) > k > 0, k odd
    freq_kernel = fp.fft2(fp.ifftshift(psf))
    return np.abs(fp.ifft2(freq*freq_kernel))

def plot_freq_filter(F, title, size=20):
    plt.imshow(20*np.log10( 0.01 + np.abs(fp.fftshift(F))), cmap='coolwarm'), plt.title(title, size=size), plt.colorbar()

def plot_freq_spec_3d(freq):
    fig = plt.figure(figsize=(10,10))
    ax = fig.gca(projection='3d')
    Y = np.arange(-freq.shape[0]//2,freq.shape[0]-freq.shape[0]//2)
    X = np.arange(-freq.shape[1]//2,freq.shape[1]-freq.shape[1]//2)
    X, Y = np.meshgrid(X, Y)
    Z = (20*np.log10( 0.01 + fp.fftshift(freq))).real
    surf = ax.plot_surface(X, Y, Z, cmap=plt.cm.coolwarm,
                           linewidth=0, antialiased=True)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    plt.show()

im = color.rgb2gray(imread('repo_file/kimi.png'))
k = 5
psf = np.ones((k, k)) / k**2
im1 = convolve2d(im, psf, k)
im1 += 0.3 * im.std() * np.random.standard_normal(im.shape)
im2, _ = restoration.unsupervised_wiener(im1, psf)
im3 = restoration.wiener(im1, psf, balance=0.25)
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 12), sharex=True, sharey=True)
plt.gray()
axes[0,0].imshow(im), axes[0,0].axis('off'), axes[0,0].set_title('Original image', size=20)
axes[0,1].imshow(im1), axes[0,1].axis('off'), axes[0,1].set_title('Noisy blurred image: PSNR={:.3f}'.format(compare_psnr(im, im1)), size=20)
axes[1,0].imshow(im2), axes[1,0].axis('off'), axes[1,0].set_title('Self tuned Wiener restoration: PSNR={:.3f}'.format(compare_psnr(im, im2)), size=20)
axes[1,1].imshow(im2), axes[1,1].axis('off'), axes[1,1].set_title('Wiener restoration: PSNR={:.3f}'.format(compare_psnr(im, im3)), size=20)
fig.tight_layout()
plt.show()

