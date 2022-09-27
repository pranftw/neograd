import _setup
import neograd as ng
import numpy as np
from neograd.nn.utils import fn_grad_check

inputs = ng.tensor(np.random.randn(4,3,6,6), requires_grad=True)
conv = ng.nn.Conv3D(3,2,(3,3))

def sample(inputs):
  outputs = conv(inputs)
  return outputs

fn_grad_check(sample, inputs, inputs, conv.kernel, conv.bias)
