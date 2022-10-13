import dill
from .model import Model


def save_model(fpath, model):
  '''Saves the model

  Saves the model by pickling it onto a file

  Args:
    fpath (str): Path in which to save the model
    model (Model): Model to be saved
  
  Raises:
    TypeError: if model isn't an instance of Model
  '''
  if not isinstance(model, Model):
    raise TypeError(f'Expected Model object, instead got {type(model)}')
  with open(fpath,'wb') as fp:
    dill.dump(model, fp)
    print(f'MODEL SAVED at {fpath}')


def load_model(fpath):
  '''Loads the model

  Args:
    fpath (str): Path from which to load the model
  
  Returns:
    Model object that is loaded
  '''
  with open(fpath,'rb') as fp:
    model = dill.load(fp)
    print(f'MODEL LOADED from {fpath}')
  return model


def get_batches(inputs, targets=None, batch_size=None):
  '''Returns batches of inputs and targets

  Split the inputs and their corresponding targets into batches for efficient
  training

  Args:
    inputs (Tensor): Inputs to be batched
    targets (Tensor): Targets to be batched. Defaults to None
    batch_size (int): Size of the batches. Defaults to None meaning batch_size
      will be same as number of examples
  
  Yields:
    Batches of inputs and their corresponding targets

  Raises:
    AssertionError: If first dimensions of inputs and targets don't match
    ValueError: If batch_size is greater than number of examples
    ValueError: If batch_size is negative
    ValueError: If batch_size is 0
  '''
  if targets is not None:
    assert inputs.shape[0]==targets.shape[0], '0th dim should be number of examples and should match'

  num_examples = inputs.shape[0]

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
  while start<num_examples:
    end = start+batch_size if start+batch_size<num_examples else num_examples
    if targets is not None:
      yield inputs[start:end], targets[start:end]
    else:
      yield inputs[start:end]
    start = end