import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import numpy as np
import argparse
import imutils
import cv2
import math
import scipy.sparse as sp
import scipy.sparse.linalg as spln

image = cv2.imread("/Users/ClaudiaEspinoza/Desktop/Patter Recognition/duck.jpg")
not_ducks = cv2.imread("/Users/ClaudiaEspinoza/Desktop/Patter Recognition/not_ducks.jpg")
ducks = cv2.imread("/Users/ClaudiaEspinoza/Desktop/Patter Recognition/is_ducks.jpg")

data_color = np.array(image)
print(data_color.shape)

gaussian_data_w0 = np.empty((0, 3))
gaussian_data_w1 = np.empty((0, 3))

not_ducks = np.array(not_ducks)
ducks = np.array(ducks)

gaussian_data_w1 = not_ducks.reshape(len(not_ducks)*len(not_ducks[0]),3)
gaussian_data_w0 = ducks.reshape(len(ducks)*len(ducks[0]),3)

#Gaussian distribution
mean_w0 = np.mean(gaussian_data_w0, axis = 0)
mean_w1 = np.mean(gaussian_data_w1, axis = 0)
cov_w0 = np.cov(gaussian_data_w0.T)
cov_w1 = np.cov(gaussian_data_w1.T)
print(mean_w0)
print(cov_w0)
print(mean_w1)
print(cov_w1)
w0 = multivariate_normal(mean_w0, cov_w0)
w1 = multivariate_normal(mean_w1, cov_w1)

for i in range(len(data_color)):
	for j in range(len(data_color[i])):
		if w0.pdf(data_color[i][j]) - w1.pdf(data_color[i][j]) > 0:
			data_color[i][j] = [255,255,255]
			continue
		data_color[i][j] = [0,0,0]
	print(i)

cv2.imwrite('output.png',data_color)
#f = np.exp(-np.square(x-mean)/2*variance)/(np.sqrt(2*np.pi*variance))
