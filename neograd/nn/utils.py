import numpy as np
from ..autograd.utils import no_track

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

def grad_check(model, inputs, targets, loss_fn):
  epsilon = 1e-7
  params = model.get_params()
  analytical_grads = []
  calculated_grads = []

  for i,param in enumerate(params):
    param.zero_grad()
    for idx in np.ndindex(param.shape):
      with no_track():
        param.data[idx]+=epsilon
        outputs = model(inputs)
        loss1 = loss_fn(outputs, targets)

        param.data[idx]-=(2*epsilon)
        outputs = model(inputs)
        loss2 = loss_fn(outputs, targets)

      param.data[idx]+=epsilon
      outputs = model(inputs)
      loss = loss_fn(outputs, targets)
      loss.backward()

      if not(isinstance(param.grad, np.ndarray)):
        param.grad = np.array(param.grad)
      calculated_grads.append(param.grad[idx])
      analytical_grads.append((loss1.data-loss2.data)/(2*epsilon))
      param.zero_grad()

  analytical_grads = np.array(analytical_grads)
  calculated_grads = np.array(calculated_grads)
  checker = np.linalg.norm(analytical_grads-calculated_grads)/(np.linalg.norm(analytical_grads) + np.linalg.norm(calculated_grads))
  print(checker)