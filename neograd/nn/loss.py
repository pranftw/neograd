from ..autograd import sum as _sum, log


class Loss:
  def __call__(self, outputs, targets):
    return self.forward(outputs, targets)
  
  def get_num_examples(self, outputs_shape):
    if len(outputs_shape)==0:
      return 1
    else:
      return outputs_shape[0] 


class MSE(Loss):
  '''
    Mean Squared Error
  '''
  def forward(self, outputs, targets):
    num_examples = self.get_num_examples(outputs.shape)
    cost = (1/(2*num_examples))*_sum((outputs-targets)**2)
    return cost
  
  def __repr__(self):
    return f'MSE()'
  
  def __str__(self):
    return 'MeanSquaredError'


class BCE(Loss):
  '''
    Binary Cross Entropy

    epsilon used here to prevent log0
  '''
  def forward(self, outputs, targets):
    epsilon = 1e-8
    num_examples = self.get_num_examples(outputs.shape)
    entropy = _sum((outputs*log(targets+epsilon)) + ((1-outputs)*(log(1-targets+epsilon))))
    cost = (-1/num_examples)*entropy
    return cost
  
  def __repr__(self):
    return f'BCE()'
  
  def __str__(self):
    return 'BinaryCrossEntropy'


class CE(Loss):
  '''
    Cross Entropy
  '''

  def forward(self, outputs, targets):
    epsilon = 1e-8
    num_examples = self.get_num_examples(outputs.shape)
    entropy = _sum(targets*log(outputs+epsilon))
    cost = (-1/num_examples)*entropy
    return cost
  
  def __repr__(self):
    return 'CE()'
  
  def __str__(self):
    return 'CrossEntropy'