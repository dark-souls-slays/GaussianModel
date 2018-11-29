<b>Claudia Espinoza<br>
National Dong Hwa Univerisity <br>
Machine Learning </b>


- Every pixel has 2 probabilities "it is part of a duck" and "it is not part of a duck".
- We assume that P(w1) and P(w2) are the same; therefore P(wi|x) = P(x|wi)


<b>Getting the value for the Gaussian Distribution<br>
1. Cut the image, so that only a small portion will be used (because of the high space complexity).
2. Change this <span style="color: blue">image_cut</span> to greyscale and perform a series of erotions and dilations to identify the bright areas.

image_cut                                          thresh


3. <span style="color: blue">Thresh</span> is cross checked with <span style="color: blue">image_cut</span> to get the RGB values of pixels forming part of the ducks.


Gaussian Distribution

The normal distribution is a form presenting data by arranging the
probability distribution of each value in the data.Most values remain
around the mean value making the arrangement symmetric.

We use various functions in numpy library to mathematically calculate
the values for a normal distribution.
