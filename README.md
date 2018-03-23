# Linear Regression

This code uses a linear regression algorithm for stock market predictions, using both the sklearn library (linear_regression1.py), and also uses linear regression for predicting the y value of arbitrary 2 dimensional data by making the algorithm from scratch (linear_regression2.py).

The stock data is aquired from the api of:
https://www.quandl.com/

The linear regression algorithm predicts data values by attempting to create a linear line of best fit by calculating the slope and y intercept using standard statistical equations or through gradient descent.

The stock predictions of linear_regression are wrong because the features given to the data set are of only the last days HLOCV price data and making the model was soley for the purpose of learning how to use sklearm for linear regression.

The predictions of the arbitrary 2 dimensional data of the model written from scratch were much more accurate. The data, along with the models line of best fit are shown below.

![](https://github.com/PopeyedLocket/linear-regression/blob/master/linear_regression2_data_and_best_fit_line.png)

This line of best fit was calculated using the standard statistical equations for finding the line of best fit:
	m = (np.mean(xs)*np.mean(ys) - np.mean(xs*ys)) / (np.mean(xs)**2 - np.mean(xs**2))
	b = np.mean(ys) - m * np.mean(xs)

Theses values are specified in the program's terminal output.
![](https://github.com/PopeyedLocket/linear-regression/blob/master/linear_regression2_output.png)


Source:
https://www.youtube.com/watch?v=Kpxwl2u-Wgk

