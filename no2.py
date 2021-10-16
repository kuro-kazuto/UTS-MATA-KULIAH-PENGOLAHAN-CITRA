# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 09:22:59 2021

@author: galih
UTS PENGCIT
Soal no 2 (Projective transform untuk gambar)
"""

from skimage.transform import ProjectiveTransform
from skimage.io import imread
import numpy as np
import matplotlib.pylab as plt

im_src = imread('repo_file/background.jpg')
im_dst = imread('repo_file/graffiti.png')
im_mask = imread('repo_file/graffiti_mask.png')
im_dst1 = np.copy(im_dst)
height, width, dim = im_src.shape
print(height, width, im_src.shape, im_dst.shape)

pt = ProjectiveTransform()
src = np.array([[   0.,    0.],
       [height-1,    0.],
       [height-1,  width-1],
       [   0.,  width-1]])
dst = np.array([[ 0., 0.],
       [im_dst.shape[0]-1, 0],
       [im_dst.shape[0]-1,  687],
       [ 0., 659]])
print(pt.estimate(src, dst))

im_dst_masked = im_dst & im_mask

x, y = np.mgrid[:im_dst.shape[0], :im_dst.shape[1]]
dst_indices = np.hstack((x.reshape(-1, 1), y.reshape(-1,1))) 
src_indices = np.round(pt.inverse(dst_indices), 0).astype(int)
valid_idx = np.where((src_indices[:,0] < height) & (src_indices[:,1] < width) & (src_indices[:,0] >= 0) & (src_indices[:,1] >= 0))
dst_indicies_valid = dst_indices[valid_idx]
src_indicies_valid = src_indices[valid_idx]
im_dst[dst_indicies_valid[:,0],dst_indicies_valid[:,1]] = im_src[src_indicies_valid[:,0],src_indicies_valid[:,1]]
im_dst &= (~im_mask) 
im_dst += im_dst_masked
plt.figure(figsize=(20,30))
plt.subplot(311), plt.imshow(im_src), plt.axis('off'), plt.title('Gambar Asal', size=30)
plt.subplot(312), plt.imshow(im_dst1), plt.axis('off'), plt.title('Background Yang Akan Diganti', size=30)
plt.subplot(313), plt.imshow(im_dst), plt.axis('off'), plt.title('Gambar Setelah Transformasi', size=30)
plt.show()

