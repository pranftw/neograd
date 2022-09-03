from autograd import sum, log


class Loss:
  def __call__(self, outputs, targets):
    return self.forward(outputs, targets)


class MSE(Loss):
  def forward(self, outputs, targets):
    num_examples = outputs.shape[-1]
    cost = (1/(2*num_examples))*sum((outputs-targets)**2)
    return cost


class BinaryCrossEntropy(Loss):
  def forward(self, outputs, targets):
    num_examples = outputs.shape[-1]
    entropy = ((outputs*log(targets)) + ((1-outputs)*(log(1-targets))))
    cost = (-1/num_examples)*(sum(entropy))
    return cost