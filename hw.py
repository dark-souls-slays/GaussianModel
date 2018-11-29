# import the necessary packages
import matplotlib.pyplot as plt
from imutils import contours
from skimage import measure
import numpy as np
import argparse
import imutils
import cv2
import math
import scipy.sparse as sp
import scipy.sparse.linalg as spln


def lognormpdf(x,mu,S):
    """ Calculate gaussian probability density of x, when x ~ N(mu,sigma) """
    nx = len(S)
    norm_coeff = nx * math.log(2 * math.pi) + np.linalg.slogdet(S)[1]

    err = x-mu
    if (sp.issparse(S)):
        numerator = spln.spsolve(S, err).T.dot(err)
    else:
        numerator = np.linalg.solve(S, err).T.dot(err)

    return -0.5*(norm_coeff+numerator)


#load the image, convert it to grayscale, and blur it
image = cv2.imread("/Users/ClaudiaEspinoza/Desktop/Patter Recognition/duck.jpg")
not_ducks = cv2.imread("/Users/ClaudiaEspinoza/Desktop/Patter Recognition/not_ducks.jpg")
image_cut = image[7200:-5400, 2000:-2000]
image = image[6000:-4600, :-1000]
cv2.imwrite('cut_try.png',image_cut)
cv2.imwrite('working_on.png',image)
gray = cv2.cvtColor(image_cut, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (11, 11), 0)

# threshold the image to reveal light regions in the blurred image
thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)[1]

# perform a series of erosions and dilations to remove
# any small blobs of noise from the thresholded image
thresh = cv2.erode(thresh, None, iterations=2)
thresh = cv2.dilate(thresh, None, iterations=4)
cv2.imwrite('thresh.png',thresh)

np.set_printoptions(threshold=np.inf)
data = np.array(thresh)
data_color = np.array(image)
data_color_cut = np.array(image_cut)
print(data.shape)
print(data_color.shape)

gaussian_data_w0 = np.empty((0, 3))
gaussian_data_w1 = np.empty((0, 3))

for i in range(len(data)):
	for j in range(len(data[i])):
		if data[i][j] == 255:
			gaussian_data_w0 = np.append(gaussian_data_w0, [data_color_cut[i][j]], axis = 0)
		#else:
		#	gaussian_data_w1 = np.append(gaussian_data_w1, [data_color_cut[i][j]], axis = 0)
        #print(data_color_cut[i][j])
	print(i)

not_ducks = np.array(not_ducks)
print(not_ducks.shape)
print(len(not_ducks))
print(len(not_ducks[0]))
gaussian_data_w1 = not_ducks.reshape(len(not_ducks)*len(not_ducks[0]),3)
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
		w0 = lognormpdf(data_color[i][j], mean_w0, cov_w0)
		w1 = lognormpdf(data_color[i][j], mean_w1, cov_w1)
		if(w0>w1):
			data_color[i][j] = [255,0,0]
        else:
            data_color[i][j] = [0,0,0]
        print(i)

cv2.imwrite('output3.png',data_color)


#f = np.exp(-np.square(x-mean)/2*variance)/(np.sqrt(2*np.pi*variance))
