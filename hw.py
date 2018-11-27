# import the necessary packages
from imutils import contours
from skimage import measure
import numpy as np
import argparse
import imutils
import cv2


# load the image, convert it to grayscale, and blur it
image = cv2.imread("/Users/ClaudiaEspinoza/Desktop/Patter Recognition/duck.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (11, 11), 0)

# threshold the image to reveal light regions in the
# blurred image
thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)[1]

# perform a series of erosions and dilations to remove
# any small blobs of noise from the thresholded image
thresh = cv2.erode(thresh, None, iterations=2)
thresh = cv2.dilate(thresh, None, iterations=4)
cv2.imwrite('thresh.png',thresh)

np.set_printoptions(threshold=np.inf)
data = np.array(thresh)
data_color = np.array(image)
print(data.shape)
print(data_color.shape)

gaussian_data_w0 = np.empty((0, 3))
gaussian_data_w1 = np.empty((0, 3))

for i in range(10):
	for j in range(5946):
		if data[i][j] == 255:
			gaussian_data_w0 = np.append(gaussian_data_w0, [data_color[i][j]], axis = 0)
		else:
			gaussian_data_w1 = np.append(gaussian_data_w1, [data_color[i][j]], axis = 0)
	print(i)


#Gaussian distribution
