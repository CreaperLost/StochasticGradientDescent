import numpy as np
from random import shuffle

def f_pred(weights,features):
  return weights[:-1].dot(features) + weights[-1]


def compute_gradient_and_loss(W, X, y, reg, reg_type, opt):
  """
  loss and gradient function.

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength
  - reg_type: (int) regularization type (1: l1, 2: l2)
  - opt: (int) 0 for computing both loss and gradient, 1 for computing loss only
  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  
  dW = np.zeros(W.shape) # initialize the gradient as zero
  
  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  
  #############################################################################
  # TODO:                                                                     #
  # Implement the routine to compute the loss, storing the result in loss     #
  #############################################################################
  W_T = np.transpose(W)
  for class_index in range(num_classes):
      print(W_T[class_index])
      for sample_index in range(num_train):
        f_pred(W_T[class_index],X[sample_index]) - y[sample_index]
  
  if opt == 0 :
    i = 0 
    print(W.shape)
    print(X.shape)
    print(y.shape)
  else :
    return loss
  print("Test")
  #############################################################################
  # TODO:                                                                     #
  # Implement the gradient for the required loss, storing the result in dW.	  #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  

  pass

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  return loss, dW