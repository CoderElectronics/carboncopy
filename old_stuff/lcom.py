import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.util import random_noise
from skimage import feature
from scipy import ndimage
from skimage.color import rgb2gray
from skimage import data
from skimage.filters import gaussian
from skimage.segmentation import active_contour
from PIL import Image

# Generate noisy image of a square
image = np.zeros((128, 128), dtype=float)
image[32:-32, 32:-32] = 1

image = ndi.rotate(image, 15, mode='constant')
image = ndi.gaussian_filter(image, 4)
image = random_noise(image, mode='speckle', mean=0.1)

# Compute the Canny filter for two values of sigma
edges = feature.canny(image, sigma=3)
edg8 = np.array(edges).astype(np.uint8)

"""
s = np.linspace(0, 2 * np.pi, 400)
r = 60 + 100 * np.sin(s)
c = 60 + 100 * np.cos(s)
init = np.array([r, c]).T

snake = active_contour(
    gaussian(edg8, sigma=3, preserve_range=False),
    init,
    alpha=0.015,
    beta=10,
    gamma=0.001,
)

fig, ax = plt.subplots(figsize=(7, 7))
ax.imshow(edg8, cmap=plt.cm.gray)
ax.plot(init[:, 1], init[:, 0], '--r', lw=3)
ax.plot(snake[:, 1], snake[:, 0], '-b', lw=3)
ax.set_xticks([]), ax.set_yticks([])
ax.axis([0, edg8.shape[1], edg8.shape[0], 0])

print("Img shape: {}".format(image.shape))
print("Edges shape: {}".format(edges.shape))
print("Edg8 shape: {}".format(edg8.shape))

#plt.imshow(edg8, cmap=plt.cm.gray)
"""

img = Image.open("demo1.jpg")

im_gray = np.array(img.convert('L'))
im_bool = im_gray > 160
im_bin = (im_gray > 160) * 255

print(im_bin)

r = np.linspace(0, im_bin.shape[0], 100)
c = np.linspace(0, im_bin.shape[1], 100)

#init = np.array([[0, 0], [0, im_bin.shape[1]], [im_bin.shape[0], 0], [im_bin.shape[0], im_bin.shape[1]]]).T

s = np.linspace(0, 2 * np.pi, 400)
r = 100 + 100 * np.sin(s)
c = 220 + 100 * np.cos(s)
init = np.array([r, c]).T

snake = active_contour(
    gaussian(im_bin, sigma=3, preserve_range=False),
    init,
    alpha=0.1,
    beta=1,
    gamma=0.1,
    w_line=-5,
    w_edge=0,
)

fig, ax = plt.subplots(figsize=(7, 7))
ax.imshow(im_bin, cmap=plt.cm.gray)
ax.plot(init[:, 1], init[:, 0], '--r', lw=3)
ax.plot(snake[:, 1], snake[:, 0], '-b', lw=3)
ax.set_xticks([]), ax.set_yticks([])
ax.axis([0, im_bin.shape[1], im_bin.shape[0], 0])

plt.show()

plt.show()

def lcom_filter(edges2, m=64, n=64):
    cmp_img = np.zeros(edges2.shape, dtype=float)

    for i in range(m):
        for j in range(n):
            y0 = i * int(edges2.shape[0] / m)
            y1 = y0 + int(edges2.shape[0] / m)
            x0 = j * int(edges2.shape[1] / n)
            x1 = x0 + int(edges2.shape[1] / n)

            tile = (edges2[y0:y1, x0:x1])
            tile_asint = np.array(tile).astype(np.uint8)

            cm = ndimage.center_of_mass(tile_asint)

            if not any(np.isnan(a) for a in cm):
                x_c = int(np.rint(cm[0]) + x0)
                y_c = int(np.rint(cm[1]) + y0)
                cmp_img[x_c, y_c] = 1

    return cmp_img


# display results
fig, ax = plt.subplots(nrows=1, ncols=10, figsize=(8, 3))

ax[0].imshow(image, cmap='gray')
ax[0].set_title('noisy image', fontsize=20)

ax[1].imshow(edges, cmap='gray')
ax[1].set_title(r'Canny filter, $\sigma=3$', fontsize=20)

for i in range(2, 9):
    lcom_img = lcom_filter(edges, m=10*i, n=10*i)

    ax[i].imshow(lcom_img, cmap='gray')
    ax[i].set_title('lcom image m/n={}'.format(10*i), fontsize=20)

for a in ax:
    a.axis('off')

fig.tight_layout()
plt.show()