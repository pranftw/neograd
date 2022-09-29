def get_batches(inputs, targets, batch_size=None):
  '''
    Split the inputs and their corresponding targets into batches for efficient
      training
    0th dim is taken as number of examples in the data
  '''
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
    yield inputs[start:end], targets[start:end]
    start = end