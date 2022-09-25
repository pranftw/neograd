import _setup
import neograd as ng
import numpy as np
from neograd.nn.loss import MSE

inputs = ng.tensor(np.random.randn(2,6,6), requires_grad=True)
conv = ng.nn.Conv2D((3,3),2,2)
loss_fn = MSE()

outputs = conv(inputs)
targets = ng.tensor(np.ones(outputs.shape))
loss = loss_fn(outputs, targets)
loss.backward()
epsilon = 1e-7

analytical = []
calculated = []

for idx in np.ndindex(inputs.shape):
  with ng.no_track():
    inputs.data[idx]+=epsilon
    outputs = conv(inputs)
    loss1 = loss_fn(outputs, targets)
    inputs.data[idx]-=(2*epsilon)
    outputs = conv(inputs)
    loss2 = loss_fn(outputs, targets)
    inputs.data[idx]+=epsilon
  calculated.append(inputs.grad[idx])
  analytical.append((loss1.data-loss2.data)/(2*epsilon))

for idx in np.ndindex(conv.kernel.shape):
  with ng.no_track():
    conv.kernel.data[idx]+=epsilon
    outputs = conv(inputs)
    loss1 = loss_fn(outputs, targets)
    conv.kernel.data[idx]-=(2*epsilon)
    outputs = conv(inputs)
    loss2 = loss_fn(outputs, targets)
    conv.kernel.data[idx]+=epsilon
  calculated.append(conv.kernel.grad[idx])
  analytical.append((loss1.data-loss2.data)/(2*epsilon))

analytical_grads = np.array(analytical)
calculated_grads = np.array(calculated)
dist = np.linalg.norm(analytical_grads-calculated_grads)/(np.linalg.norm(analytical_grads) + np.linalg.norm(calculated_grads))
print(dist)