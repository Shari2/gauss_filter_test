#! /usr/bin/env python3

import cv2
import numpy as np
import matplotlib.pylab as plt
import scipy.signal as signal
import scipy.stats as stats
import math

from scipy.ndimage import gaussian_filter1d

# img = cv2.imread("Data2.bmp", cv2.IMREAD_GRAYSCALE)
# y,x = img.shape
# # pixel coords img[0,0] is top left corner
# # pixel coords img[y - 1, x - 1] is bottom right corner
# 
# #cv2.imshow("Import", img)
# #cv2.waitKey(0)
# #cv2.destroyAllWindows()
# 
# 
# 
# y_array = np.array([])
# 
# 
# # left to right
# for i in range(x):
#   # top to bottom
#   for j in range(y):
#     if img[j, i] < 255:
#       y_array = np.append(y_array, [y - 1 -j])
#       break
# 
# 
# np.savetxt("Data2.csv", y_array, delimiter=",")

y_array = np.loadtxt("Data2.csv", delimiter=",")

x = y_array.shape[0]

fig = plt.figure(1)


x_array = np.linspace(0, x - 1, x)

# max value is 1.0
sig = y_array/np.max(y_array)

#sig = np.sin(x_array * np.pi / 180)


lut = [ 0, 2, 3, 5, 7, 11, 17, 23, 29, 37, 53, 71, 97, 127, 157, 197]

max_plots = len(lut)

org = fig.add_subplot(max_plots, 1, 1)
org.plot(sig)
#org.margins(0, 0.1)
org.set_title('Original')

for i in range(2, max_plots, 1):
  sigma = lut[i]

  filtered = gaussian_filter1d(sig, sigma, order=0)
  ax = fig.add_subplot(max_plots, 1, i, sharex=org)
  ax.plot(filtered)
  #ax.margins(0, 0.1)
  ax.set_title('Sigma = {}'.format(sigma))
  ax.get_xaxis().set_visible(i == max_plots - 1)



#fig.tight_layout()
fig.show()
plt.show()


