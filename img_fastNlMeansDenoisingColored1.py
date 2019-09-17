import numpy as np
import cv2
from matplotlib import pyplot as plt
img = cv2.imread('Image 6.JPG')# read the image
dst = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)#apply the denoising method
plt.subplot(121),plt.imshow(img)#show the original image
plt.subplot(122),plt.imshow(dst)#show the filtered image
plt.show()

