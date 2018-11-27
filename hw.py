# import the necessary packages
import matplotlib.pyplot as plt
from imutils import contours
from skimage import measure
import numpy as np
import argparse
import imutils
import cv2
import math


def norm_pdf_multivariate(x, mu, sigma):
    size = len(x)
    if size == len(mu) and (size, size) == sigma.shape:
        det = linalg.det(sigma)
        if det == 0:
            raise NameError("The covariance matrix can't be singular")

        norm_const = 1.0/ ( math.pow((2*pi),float(size)/2) * math.pow(det,1.0/2) )
        x_mu = matrix(x - mu)
        inv = sigma.I
        result = math.pow(math.e, -0.5 * (x_mu * inv * x_mu.T))
        return norm_const * result
    else:
        raise NameError("The dimensions of the input don't match")

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
cv2.imwrite('thresh_try.png',thresh)

np.set_printoptions(threshold=np.inf)
data = np.array(thresh)
data_color = np.array(image)
print(data.shape)
print(data_color.shape)

gaussian_data_w0 = np.empty((0, 3))
gaussian_data_w1 = np.empty((0, 3))

for i in range(len(data)):
	for j in range(len(data[i])):
		if data[i][j] == 255:
			gaussian_data_w0 = np.append(gaussian_data_w0, [data_color[i][j]], axis = 0)
		else:
			gaussian_data_w1 = np.append(gaussian_data_w1, [data_color[i][j]], axis = 0)
	print(i)


#Gaussian distribution

mean_w0 = np.mean(gaussian_data_w0, axis = 0)
mean_w1 = np.mean(gaussian_data_w1, axis = 0)
cov_w0 = np.cov(gaussian_data_w0.T)
cov_w1 = np.cov(gaussian_data_w1.T)
print(mean_w0)
print(cov_w0)
print(mean_w1)
print(cov_w1)

for i in range(len(data_color)):
	for j in range(len(data_color[i])):
		w0 = norm_pdf_multivariate(data_color[i][j], mean_w0, cov_w0)
		w1 = norm_pdf_multivariate(data_color[i][j], mean_w1, cov_w1)
		if(w0>w1):
			data_color[i][j] = [255,0,0]

cv2.imwrite('output_try.png',data_color)


#f = np.exp(-np.square(x-mean)/2*variance)/(np.sqrt(2*np.pi*variance))
