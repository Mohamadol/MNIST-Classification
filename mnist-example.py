#openml.org is a public repository for machine learning
from sklearn.datasets import fetch_openml   
import matplotlib as mpl 
import matplotlib.pyplot as plt 
import numpy as np
from sklearn.linear_model import SGDClassifier

mnist = fetch_openml('mnist_784', version=1)
keys = mnist.keys()

for key in keys:
    print (key)

#X is a 70,000 x 784 matrix of samples and features 
#features are the 28x28 pixels of each image
#y is a 70,000 x 1 vector of samples
X, y = mnist['data'], mnist['target']

#randomly chose the first element of dataset for demonstration
sample_digit = X[0]	
#reshape it to an image
sample_digit_image = sample_digit.reshape(28, 28)

plt.imshow(sample_digit_image, cmap=mpl.cm.binary, interpolation="nearest")
plt.axis("off")
#the image is digit 5 and the y[0] value will confirm it
plt.show() 

#cast the labels to integers
y = y.astype(np.uint8) 
#re-structing datasets and splitting into training and test
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

#binary classifier that recognizes digit 5
y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)
sgd_clf = SGDClassifier(random_state=42) #to get the same parameters everytime
sgd_clf.fit(X_train, y_train_5) 




