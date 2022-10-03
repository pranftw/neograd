from ..autograd import sum as _sum, log


class Loss:
  '''Base class of all loss functions
  '''
  def __call__(self, outputs, targets):
    '''Abstracts the forward method

    Args:
      outputs (Tensor): Outputs of Layer/Container/Model/Operation
      targets (Tensor): Targets to be evaluated against
    
    Returns:
      Tensor of the result
    '''
    return self.forward(outputs, targets)
  
  def get_num_examples(self, outputs_shape):
    '''Gathers the number of examples

    If dimensions of outputs_shape is 0, ie it is a scalar, then num_examples is 1
    Else the first dimension value of outputs_shape is taken as num_examples

    Args:
      outputs_shape (tuple of int): Shape of the outputs
    
    Returns:
      Number of examples
    '''
    if len(outputs_shape)==0:
      return 1
    else:
      return outputs_shape[0] 


class MSE(Loss):
  '''Mean Squared Error
  '''
  def forward(self, outputs, targets):
    '''Forward pass of MSE

    Args:
      outputs (Tensor): Outputs of Layer/Container/Model/Operation
      targets (Tensor): Targets to be evaluated against
    
    Returns:
      Tensor of the result
    '''
    num_examples = self.get_num_examples(outputs.shape)
    cost = (1/(2*num_examples))*_sum((outputs-targets)**2)
    return cost
  
  def __repr__(self):
    return f'MSE()'
  
  def __str__(self):
    return 'MeanSquaredError'


class BCE(Loss):
  '''Binary Cross Entropy
  '''
  def forward(self, outputs, targets, epsilon=1e-8):
    '''Forward pass of BCE

    epsilon used  to prevent log0

    Args:
      outputs (Tensor): Outputs of Layer/Container/Model/Operation
      targets (Tensor): Targets to be evaluated against
    
    Returns:
      Tensor of the result
    '''
    num_examples = self.get_num_examples(outputs.shape)
    entropy = _sum((outputs*log(targets+epsilon)) + ((1-outputs)*(log(1-targets+epsilon))))
    cost = (-1/num_examples)*entropy
    return cost
  
  def __repr__(self):
    return f'BCE()'
  
  def __str__(self):
    return 'BinaryCrossEntropy'


class CE(Loss):
  '''Cross Entropy
  '''
  def forward(self, outputs, targets, epsilon=1e-8):
    '''Forward pass of CE

    epsilon used  to prevent log0

    Args:
      outputs (Tensor): Outputs of Layer/Container/Model/Operation
      targets (Tensor): Targets to be evaluated against
    
    Returns:
      Tensor of the result
    '''
    num_examples = self.get_num_examples(outputs.shape)
    entropy = _sum(targets*log(outputs+epsilon))
    cost = (-1/num_examples)*entropy
    return cost
  
  def __repr__(self):
    return 'CE()'
  
  def __str__(self):
    return 'CrossEntropy'