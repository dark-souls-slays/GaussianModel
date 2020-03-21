<b>Claudia Espinoza<br>
National Dong Hwa Univerisity <br>
Machine Learning </b>


- Every pixel has 2 probabilities "it is part of a duck" and "it is not part of a duck".
- We assume that P(w1) and P(w2) are the same; therefore P(wi|x) = P(x|wi)

Our example is a trivial way to demonstrate the Bayesian Theorem, given that we
only need the likelihood to estimate the posterior probability.

original image<br>
![alt text](https://github.com/dark-souls-slays/GaussianModel/blob/master/duck.jpg)

<b>Getting the value for the Gaussian Distribution<br></b>
1. Collect areas of the picture where ducks don't appear and merge them into a single image
2. Collect duck bodies pixels and merge them into a single image

is_ducks<br>                                          
![alt text](https://github.com/dark-souls-slays/GaussianModel/blob/master/is_ducks.jpg)
not_ducks<br>
![alt text](https://github.com/dark-souls-slays/GaussianModel/blob/master/not_ducks.jpg)

3. The values of this pixels are used to find the mean and covariance matrix of w0 and w1.


<b>Gaussian Distribution</b>

The normal probability distribution is a form of presenting data by arranging the
probability distribution of each value in the data. Most values remain
around the mean value making the arrangement symmetric.

```python
gaussian_data_w1 = not_ducks.reshape(len(not_ducks)*len(not_ducks[0]),3)
gaussian_data_w0 = ducks.reshape(len(ducks)*len(ducks[0]),3)

mean_w0 = np.mean(gaussian_data_w0, axis = 0)
mean_w1 = np.mean(gaussian_data_w1, axis = 0)
cov_w0 = np.cov(gaussian_data_w0.T)
cov_w1 = np.cov(gaussian_data_w1.T)
```

<b>Prediction</b>

Multivariate_normal function is found in scipy.stats library.
```python
w0 = multivariate_normal(mean_w0, cov_w0)
w1 = multivariate_normal(mean_w1, cov_w1)

for i in range(len(data_color)):
	for j in range(len(data_color[i])):
		if w0.pdf(data_color[i][j]) - w1.pdf(data_color[i][j]) > 0:
			data_color[i][j] = [255,255,255]
			continue
		data_color[i][j] = [0,0,0]
```
<b>Output</b><br>
![alt text](https://github.com/dark-souls-slays/GaussianModel/blob/master/output.png)

<b>Summary</b><br>
Finding the gaussian distribution of a sample of the total image and using that
distribution to estimate whether a pixel belonged to a duck or not was a more
efficient way to solve this problem; giving that a training algorithm would have
taken a higher time complexity.

<b>Comments</b><br>
Given that we assumed the P(w0) == P(w1), we have some noise in the final
output. That is because the probability of a pixel belonging to a duck is actually
a lot less than the probability of the pixel being part of any other part of the
image (P(w0)<<P(w1)). Noise can be regulated by adding extra pixels belongin to
rocks to the "not_ducks" image used to get the PDF of w1.
Other approaches were tried, but due to the amount of pixels in the original
picture the time complexity escalates really fast. The fastest way to solve the
problem was to automatically get the matrix of values for the PDF from a picture.
This is why we take the time to manually cut small pieces of the original image
and create a collage.
