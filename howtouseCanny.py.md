# Guide to using the ellipse detector API
## This API searches for ellipses in images.
## To use it you need to instance the class "ellipseDetector" with an instance of PIL Image you can choose the dimension of the image to be processed the default is 128 (you can put test = true to limit the verbose of the class).
## there are many functions to calculate the ellipse but we recommend the use of find_median_ellipse2() function to get the best result (the parameters of the ellipse are stored in the p attribute of the class) alternatively you can use find_best_ellipse() function.
## Callind the perimeter function calculates the perimeter of the ellipse in relation to the dimensions of the image.