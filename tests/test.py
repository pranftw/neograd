import _setup
import neograd as ng
import numpy as np
from neograd.nn.loss_fns import BinaryCrossEntropy
from neograd.nn.optim import GD
from neograd.nn.utils import get_batches
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split

X, y = make_circles(n_samples=1000, noise=0.05, random_state=100)
X_train, X_test, y_train, y_test = train_test_split(X,y)

X_train, X_test = ng.tensor(X_train.T), ng.tensor(X_test.T)
y_train, y_test = ng.tensor(y_train.T.reshape(1,750)), ng.tensor(y_test.T.reshape(1,250))

num_train = 750
num_test = 250
num_iter = 1

class NN(ng.nn.Model):
  def __init__(self):
    super().__init__(self)
    self.stack = ng.nn.Sequential(
      ng.nn.Linear(2,100),
      ng.nn.ReLU(),
      ng.nn.Linear(100,1),
      ng.nn.Sigmoid()
    )
  
  def forward(self, inputs):
    return self.stack(inputs)

model = NN()
loss_fn = BinaryCrossEntropy()
optim = GD(model.get_params(), 0.05)

# train_data_gen = get_batches(X_train, y_train, num_train, 50) TODO: ISSUE WRT DIMS

for iter in range(num_iter):
  optim.zero_grad()
  outputs = model(X_train)
  loss = loss_fn(outputs, y_train)
  loss.backward()
  optim.step()
  print(f"iter {iter+1}/{num_iter}\nloss: {loss}\n")