import numpy as np

#this is a basic 2 layer neural network 


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
y = np.array([[0,0,1,1]]).T

#seed this to make this deterministic
#np.random.seed(42)
np.random.seed(1)


#initialize weights randomly with mean 0

synapse0 = 2*np.random.random((3,1)) - 1
#synapse0 = np.random.random((3,1)) - 1

#synapse0 is the first layer of weights and connects layer0 with layer1


layer0 = X

for i in xrange(10000):

  #first calculate forward propagation
  layer1 = sigmoid(np.dot(layer0,synapse0))

  #calculate error rate
  layer1_error = y - layer1

  #how much we missed times the the slope of the sigmoid 
  layer1_delta = layer1_error * sigmoid(layer1,True)

  synapse0 += np.dot(layer0.T,layer1_delta)


print "post training:"
print layer1


