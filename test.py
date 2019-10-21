# -*- coding: utf-8 -*-
"""
@File    : test
@Time    : 2019-10-05
@Author  : JiutongZhao
@Email   : jtz@pku.edu.cn
@Content :
@Output  :
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import scipy
from scipy import signal

img = cv2.imread('./barbara.png', cv2.IMREAD_GRAYSCALE)

s = 11
sigma = 10
nsr = 0.00

temp_ker = np.exp(-((np.arange(s) - (s - 1) / 2) ** 2) / (2 * sigma * sigma)).reshape(s, 1) * np.exp(
    -((np.arange(s) - (s - 1) / 2) ** 2) / (2 * sigma * sigma)).reshape(1, s)
temp_ker = temp_ker / np.sum(temp_ker)

ker = np.zeros(img.shape)
ker[:s, :s] = temp_ker
ker = ker[:, (np.arange(img.shape[0]) + s // 2) % img.shape[0]]
ker = ker[(np.arange(img.shape[1]) + s // 2) % img.shape[1]]

blurred_img = np.abs(np.fft.ifft2(np.fft.fft2(img) * np.fft.fft2(ker))) + nsr * np.max(img) * np.random.rand(
    img.shape[0], img.shape[1])

Astar_A_u = np.fft.ifft((np.abs(np.fft.fft2(ker)) ** 2) * np.fft.fft2(img))
Astar_u0 = np.fft.ifft(np.conj(np.fft.fft2(ker)) * np.fft.fft2(blurred_img))

ax1 = plt.subplot(1, 3, 1, aspect='equal')
ax2 = plt.subplot(1, 3, 2, aspect='equal')
ax3 = plt.subplot(1, 3, 3, aspect='equal')

ax1.imshow(img, cmap='gray', vmax=256, vmin=0)
ax2.imshow(blurred_img, cmap='gray', vmax=256, vmin=0)
ax3.imshow(np.abs(Astar_u0 - Astar_A_u)/(np.abs(Astar_u0 + Astar_A_u)), cmap='gray', vmax=1, vmin=0)
# ax3.imshow(blurred_img - img, cmap='gray', vmax=256, vmin=0)
# ax3.imshow(blurred_img - blurred_img_2, cmap='jet', vmax=10, vmin=-10)

ax1.axis('off')
ax2.axis('off')
ax3.axis('off')

ax1.set_title('Origin')
ax2.set_title('Blurred + Noise')
ax3.set_title('Deblurred')

plt.gcf().set_size_inches(10, 4)

plt.savefig('test.pdf')

plt.show()
