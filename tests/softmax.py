import _setup
import neograd as ng
import numpy as np
from neograd.autograd.ops import softmax
from neograd.nn.loss import CE

y = ng.tensor([[0, 0, 1],[1, 0, 0]])
loss_fn = CE()

a = ng.tensor([[1,2,3],[5,2,8]], requires_grad=True)
outputs = softmax(a,1)
loss = loss_fn(outputs, y)
loss.backward()

epsilon = 1e-7

analytical = []
calculated = []

with ng.no_track():
  for idx in np.ndindex(a.data.shape):
    a.data[idx]+=epsilon
    outputs = softmax(a,1)
    loss1 = loss_fn(outputs, y)
    a.data[idx]-=(2*epsilon)
    outputs = softmax(a,1)
    loss2 = loss_fn(outputs, y)
    a.data[idx]+=epsilon
    analytical.append((loss1.data-loss2.data)/(2*epsilon))
    calculated.append(a.grad[idx])

analytical_grads = np.array(analytical)
calculated_grads = np.array(calculated)

dist = np.linalg.norm(analytical_grads-calculated_grads)/(np.linalg.norm(analytical_grads) + np.linalg.norm(calculated_grads))
print(dist)