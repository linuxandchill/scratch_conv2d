import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
import matplotlib.image as mpimg

img = mpimg.imread('lena.jpg')
#print(img.shape)

bw = img.mean(axis=2)

'''
Hx = np.array([
    [-1,0,1],
    [-2,0,2],
    [-1,0,1]
    ], dtype=np.float32)
'''
Hx = np.array([
    [-1,0,1,2],
    [-2,0,2,2],
    [-1,0,-2,-1],
    [-3,0,1,2],
    ], dtype=np.float32)

Hy = Hx.T

#horizontal edges
Gx = convolve2d(bw, Hx)

#vertical edges
Gy = convolve2d(bw, Hy)


#plt.imshow(Gx, cmap='gray')
#plt.imshow(Gy, cmap='gray')


G = np.sqrt(Gx*Gx + Gy*Gy)
plt.imshow(G, cmap='gray')
plt.show()
