import numpy as np
from random import shuffle

#The prediction given a specific (weight,features)
def f_pred(weights,features,n_class):
  preds=np.zeros((n_class,1))
  #make prediction for each class. W_i * X + B_i
  for class_index in range(n_class):
    preds[class_index] = (weights[class_index][:-1].dot(features) + weights[class_index][-1])
  return preds

def hinge_loss(preds,label):
  # predictions of other classes.
  s_j = np.delete(preds,label,axis=1)
  # prediction of the true class.
  s_y = preds[label]
  return max(0,1+max(s_j)-s_y)

def identity_f(preds,label):
  # predictions of other classes.
  s_j = np.delete(preds,label,axis=1)
  # prediction of the true class.
  s_y = preds[label]
  return 1+max(s_j)-s_y > 0

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
  
  #Partial loss computed with out regularization.
  for sample_index in range(num_train):
    loss += hinge_loss(f_pred(W_T,X[sample_index],num_classes),y[sample_index]) 
  #Mean Loss.
  loss /= num_train

  #Compute regularazation
  regularization = 0 
  if reg_type == 1:
    regularization = np.sum(abs(W_T))*num_train
  else:
    regularization = np.sum(abs(W_T)**2)*num_train

  #Compute final loss.
  loss -= reg * regularization


  if opt == 1 :
    return loss, dW

  for sample_index in range(num_train): 
    dW += identity_f(f_pred(W_T,X[sample_index],num_classes),y[sample_index]) 

  for sample_index in range(num_train):
    

  #############################################################################
  # TODO:                                                                     #
  # Implement the gradient for the required loss, storing the result in dW.	  #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  
  

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  return loss, dW