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
y = np.array([[0,1,1,0]]).T

#seed this to make this deterministic
#np.random.seed(42)
np.random.seed(1)


#initialize weights randomly with mean 0

synapse0 = 2*np.random.random((3,4)) - 1
synapse1 = 2*np.random.random((4,1)) - 1

#synapse0 is the first layer of weights and connects layer0 with layer1


layer0 = X

for i in xrange(60000):

  #first calculate forward propagation
  layer1 = sigmoid(np.dot(layer0,synapse0))
  layer2 = sigmoid(np.dot(layer1,synapse1))

  #calculate error rate
  layer2_error = y - layer2

  if (i% 10000) == 0:
    print "Error:" + str(np.mean(np.abs(layer2_error)))

  #how much we missed times the the slope of the sigmoid 
  layer2_delta = layer2_error * sigmoid(layer2,True)


  layer1_error = layer2_delta.dot(synapse1.T)
  layer1_delta = layer1_error * sigmoid(layer1,deriv=True)

  synapse1 += layer1.T.dot(layer2_delta)
  synapse0 += layer0.T.dot(layer1_delta)


#print "post training:"
#print layer1
#print layer2


