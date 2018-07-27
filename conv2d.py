import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
import scipy
import matplotlib.image as mpimg

img = mpimg.imread('lena.jpg')
#print(img.shape)

bw = img.mean(axis=2)
#print(bw.shape)

#create gaussian filter
W = np.zeros((20,20))

for i in range(20):
    for j in range(20):
        dist = (i-9.5)**2 + (j-9.5)**2
        W[i,j] = np.exp(-dist / 50)

output = convolve2d(bw, W, mode='same')

#apply filter to 3 channels
three_chans = np.zeros(img.shape)  #false
three_chans = np.zeros(bw.shape)  #true 

out_three = np.zeros(img.shape) #(512,512,3)
for i in range(3):
    out_three[:,:,i] = convolve2d(img[:,:,i], W, mode='same')

plt.imshow(out_three)
plt.show()

#plt.imshow(output, cmap='gray')
#plt.show()

#print(output.shape)


