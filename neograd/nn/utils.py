import numpy as np
from ..autograd.utils import no_track, new_graph
from ..autograd import tensor
from .loss import MSE


def get_batches(inputs, targets, num_examples, batch_size=None):
  '''
    Split the inputs and their corresponding targets into batches for efficient
      training
    TODO: Currently only works for 1D data, have to fix it to accomodate all
      dimensional data
  '''
  if batch_size is not None:
    if batch_size > num_examples:
      raise ValueError("Batch size cannot be greater than the number of examples")
    elif batch_size<0:
      raise ValueError("Batch size cannot be negative")
    elif batch_size==0:
      raise ValueError("Batch size cannot be zero")
  else:
    batch_size = num_examples
  start = 0
  for end in range(batch_size, num_examples, batch_size):
    yield (inputs[start:end], targets[start:end])
    start = end
  if end!=num_examples-1:
    yield (inputs[end:], targets[end:])


def _evaluate_grad_check(analytical_grads, calculated_grads, epsilon):
  dist = np.linalg.norm(analytical_grads-calculated_grads)/(np.linalg.norm(analytical_grads) + np.linalg.norm(calculated_grads))
  print("Gradient Check Distance:", dist)
  if dist<epsilon:
    print("Gradient Check PASSED")
  else:
    print("Gradient Check FAILED")


def _wiggle_params(analytical_grads, calculated_grads, params, get_loss, epsilon):
  for param in params:
    if param.requires_grad:
      if not(isinstance(param.grad, np.ndarray)):
        param.grad = np.array(param.grad)
      for idx in np.ndindex(param.shape):
        with no_track():
          param.data[idx]+=epsilon # PLUS
          loss1 = get_loss()
          param.data[idx]-=(2*epsilon) # MINUS
          loss2 = get_loss()
          param.data[idx]+=epsilon # ORIGINAL
        calculated_grads.append(param.grad[idx])
        analytical_grads.append((loss1.data-loss2.data)/(2*epsilon))


def grad_check(model, inputs, targets, loss_fn, epsilon=1e-7):
  '''
    Implements Gradient Check, to make sure that backprop is calculating
      the right gradients.
    If distance between backprop gradients and numerical gradients is less
      than epsilon, then the gradients are proper, if not there is
      an issue
    
    Params:
      model:Model - The Neural Network to be evaluated
      inputs:Tensor - Input data(No need for complete data, only sample enough)
      targets:Tensor - Targets
      loss_fn:Loss - Loss Function
      epsilon:float
  '''
  params = model.get_params()
  analytical_grads = []
  calculated_grads = []

  def get_loss():
    outputs = model(inputs)
    loss = loss_fn(outputs, targets)
    return loss

  with new_graph():
    loss = get_loss()
    loss.backward()
    _wiggle_params(analytical_grads, calculated_grads, params, get_loss, epsilon)

  analytical_grads = np.array(analytical_grads)
  calculated_grads = np.array(calculated_grads)
  _evaluate_grad_check(analytical_grads, calculated_grads, epsilon)


def fn_grad_check(fn, inputs, *params, targets=None, loss_fn=None, epsilon=1e-7):
  '''
    Implements Gradient Check for a function instead of a complete model
    Any params that are required to be gradient checked can be specified
    targets default is ones and loss_fn default is MSE

    Params:
      fn - Function to be gradient checked
      inputs:Tensor - inputs to the function
      params:*(Tensor) - the params whose data can be wiggled to get the gradients
      targets:Tensor - targets of the function
      loss_fn:Loss - loss_fn to evaluate the function
      epsilon:float
  '''
  loss_fn = MSE() if loss_fn is None else loss_fn
  analytical_grads = []
  calculated_grads = []

  def get_loss(targets=targets):
    outputs = fn(inputs)
    targets = tensor(np.ones(outputs.shape)) if targets is None else targets
    loss = loss_fn(outputs, targets)
    return loss
  
  with new_graph():
    loss = get_loss()
    loss.backward()
    _wiggle_params(analytical_grads, calculated_grads, params, get_loss, epsilon)

  analytical_grads = np.array(analytical_grads)
  calculated_grads = np.array(calculated_grads)
  _evaluate_grad_check(analytical_grads, calculated_grads, epsilon)
