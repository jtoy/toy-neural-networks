import numpy as np


def sigmoid(x,deriv=False):
  if(deriv==True):
    return x*(1-x)
  return 1/(1+np.exp(-x))


#test input
X = np.array([[0,0,1],
              [0,1,1],
              [1,0,1],
              [1,1,1]])

#output dataset
y = np.array([[0,0,1,1,]]).T

#seed this
np.random.seed(42)


#initialize weights randomly with mean 0

syn0 = 2*np.random.random((3,1)) - 1


for i in xrange(10000):

  #first calculate forward propagation
  l0 = X
  l1 = sigmoid(np.dot(l0,syn0))

  #calculate error rate
  l1_error = y - l1

  l1_delta = l1_error * sigmoid(l1,True)

  syn0 += np.dot(l0.T,l1_delta)


print "post training:"
print l1


