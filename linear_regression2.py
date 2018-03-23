import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random

style.use('fivethirtyeight')



def create_data(n, variance, step, coorelation=False):
	val = 1 # val = first y value
	ys = []
	for i in range(n):
		ys.append(val + random.randrange(-variance, variance))
		if coorelation and coorelation == 'pos':
			val += step
		elif coorelation and coorelation == 'neg':
			val -= step
	xs = [i for i in range(n)]
	return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)

# "train" linear regression model
def best_fit_slope_and_y_intercept(xs, ys):

	m = (np.mean(xs)*np.mean(ys) - np.mean(xs*ys)) / \
		(np.mean(xs)**2 - np.mean(xs**2))

	b = np.mean(ys) - m * np.mean(xs)

	return m, b

def predict_y(m, x, b):

	return m * x + b

# determined confidence of linear regression model
# r^2 = coefficient of determination
# high r^2 value (close to 1.00) represents confident regression line
# low r^2 value (close to 0.00) represents unconfident regression line
# https://www.youtube.com/watch?v=-fgYp74SNtk
def r_squared(m, b, xs, ys):
	
	return 1.00 - squared_error(m, b, xs, ys) / squared_error(0.00, np.mean(ys), xs, ys)
def squared_error(m, b, xs, ys):
	squared_error = 0.0
	for i in range(len(ys)):
		x, y = xs[i], ys[i]
		squared_error += (y - (m*x+b))**2
	return squared_error






# xs = np.array([1,2,3,4,5,6], dtype=np.float64)
# ys = np.array([5,4,6,5,6,7], dtype=np.float64)
xs, ys = create_data(40, 10, 2, 'pos')

m, b = best_fit_slope_and_y_intercept(xs, ys)

print 'm = %f' % m
print 'b = %f' % b


regression_line = [(m*x+b) for x in xs]

coefficient_of_determination = r_squared(m, b, xs, ys)
print 'coefficient_of_determination = %f' % coefficient_of_determination

predicted_x = 8.0
predicted_y = predict_y(m, predicted_x, b)
print 'predicted y for x = %f is: %f' % (predicted_x, predicted_y)

plt.scatter(xs, ys)
plt.plot(xs, regression_line)
plt.show()



