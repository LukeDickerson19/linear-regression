import pandas as pandas
import quandl
import math
import numpy as np
from sklearn import preprocessing ,cross_validation, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pylab as plt


api_key = 'swGsSfeRyvbCpS6QTEqJ'

# get data
# in this case, the data is google daily stock price data
df = quandl.get('WIKI/GOOGL', api_key=api_key)

print df.head()

# get rid of useless features
df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]

# define relational features
df['HL_PCT'] = (df['Adj. High']  - df['Adj. Low'])  / df['Adj. Low']  # daily volitility
df['PCT_CH'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] # daily percent change

# isolate data we care about
df = df[['Adj. Close', 'HL_PCT', 'PCT_CH', 'Adj. Volume']]

forcast_col = 'Adj. Close'
#forcast_col = 'PCT_CH'

df.fillna(-99999, inplace=True) # cant work with NaN data, replace it with -99999 (outliner)

forcast_out = 1#int(math.ceil(0.01*len(df))) # number of days into future that we're trying to predict

df['label'] = df[forcast_col].shift(-forcast_out)


print df.head()

# identify features and labels
X = np.array(df.drop(['label'],1))
X = preprocessing.scale(X)
X = X[:-forcast_out]
X_lately = X[-forcast_out:]

df.dropna(inplace=True)

Y = np.array(df['label'])


# shuffle and split data for training and testing
test_size = 0.02 # test_size = percent of data to test with 
X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, Y, test_size=test_size)
# cutoff = int(X.shape[0]*(1.00-test_size))
# X_train, X_test = X[:cutoff], X[-(X.shape[0] - cutoff):]
# Y_train, Y_test = Y[:cutoff], Y[-(Y.shape[0] - cutoff):]

#clf = svm.SVR()
clf = LinearRegression()
clf.fit(X_train, Y_train)
accuracy = clf.score(X_test, Y_test)
print('\nAccuracy = %.2f%%' % (100*accuracy))


predicted = clf.predict(X_test)

try:
    fig = plt.figure()
    plt.plot(predicted, color='blue')
    plt.plot(Y_test, color='red')
    plt.show()
except Exception as e:
    print(str(e))