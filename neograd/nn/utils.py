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