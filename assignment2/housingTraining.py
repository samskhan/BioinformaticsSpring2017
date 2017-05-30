#Author: Sams Khan
#Description: Implementing linear regression model to train training data and predicting values
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#Computing mean
def mean(input):
    return sum(input)/float(len(input))

#Computing variance
def variance(input,mean):
    return sum([(x-mean)**2 for x in input])

#Computing covariance
def covariance(x,mean_x,y,mean_y):
    cov=0.0
    for i in range(len(x)):
        cov+=(x[i]-mean_x)*(y[i]-mean_y)
    return cov

#Computing coeffecients of each column, b0 is intercept and b1 is the slope
def coefficients(xdata, ydata):

    xMean, yMean = mean(xdata), mean(ydata)
    b1 = covariance(xdata, xMean, ydata, yMean) / variance(xdata, xMean)
    b0 = yMean - b1 * xMean
    return [b0, b1]

#Implemention of residual sum
def residual_sum():
    k = 0
    resoutput=0.0
    while (k <= 12):
        xData = pd.read_csv('housing_training.csv', usecols=[k], header=None)
        xArr = xData.as_matrix()
        b0, b1 = coefficients(xArr, yArr)
        testinput = pd.read_csv('housing_test.csv', usecols=[k], header=None)
        testinparr= testinput.as_matrix()
        testinparr=np.array(testinparr)

        for inp in testinparr:
            resoutput += ((testinparr[205]-testinparr[inp]*b1)*(testinparr[205]-testinparr[inp]*b1))
        k += 1
    return resoutput

#Calculating coeffecients and intercepts
ypred = list()
yData=pd.read_csv('housing_training.csv', usecols=[13], header=None)
yArr=yData.as_matrix()
print('Now printing the list of coeffecients B0 is the intercept and B1 is the slope')
k=0;
while(k<=12):

    xData=pd.read_csv('housing_training.csv', usecols=[k], header=None)
    xArr=xData.as_matrix()
    b0, b1 = coefficients(xArr, yArr)
    k+=1
    print('Coefficients of column %.3f : B0=%.3f, B1=%.3f' % (k, b0, b1))
    #Calculating linear regression
    p=0
    while(p<=12):
        testdata=pd.read_csv('housing_test.csv', usecols=[p], header=None)
        testArr=testdata.as_matrix()
        traindata=pd.read_csv('housing_training.csv', usecols=[p], header=None)
        trainArr=traindata.as_matrix()
        b0, b1 = coefficients(trainArr, yArr)
        for elements in testArr:
            ypred.append(b0 + b1 * testArr[0])
        p+=1
    ypredArr = np.asarray(ypred)

#plotting preditiction vs ground truth
groundtruth = pd.read_csv('housing_test.csv', usecols=[13], header=None)
truthArr= groundtruth.as_matrix()
appendtruthArr = np.concatenate((truthArr, truthArr, truthArr, truthArr, truthArr, truthArr, truthArr, truthArr, truthArr, truthArr, truthArr, truthArr, truthArr))
dupRemypred =ypredArr.flatten()

print(len(dupRemypred))
print(len(truthArr))
print(len(appendtruthArr))
print (dupRemypred)
plt.scatter(dupRemypred, appendtruthArr, c='y')
plt.title('prediction vs ground truth')
plt.xlabel("prediction", fontsize=18)
plt.ylabel('ground truth', fontsize=18)
plt.show()

#Plotting prediction vs truth with the difference of residual square implemented
dupRemypred=np.array(dupRemypred)
appendtruthArr=np.array(appendtruthArr)
kekdupRemypred[:] = [residual_sum()-x for x in dupRemypred]
kekappendtruthArr[:] = [residual_sum()-x for x in appendtruthArr]