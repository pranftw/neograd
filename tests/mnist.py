import _setup
import neograd as ng
import numpy as np
from neograd.nn.loss import CE
from neograd.nn.optim import Adam
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y)

num_train, num_test = 10, 10
num_iter = 5

def one_hot(cls_arr, num_examples, num_classes):
  encoded = np.zeros((num_examples, num_classes))
  for i in range(num_examples):
    for j in range(num_classes):
      if j==cls_arr[i]:
        encoded[i][j] = 1
  return encoded

X_train = ng.tensor(X_train[:num_train,:].reshape(num_train,8,8))
X_test = ng.tensor(X_test[:num_test,:].reshape(num_test,8,8))
y_train = ng.tensor(one_hot(y_train[:num_train], num_train, 10))
y_test = ng.tensor(one_hot(y_test[:num_test], num_test, 10))

# print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

class NN(ng.nn.Model):
  def __init__(self):
    super().__init__(self)
    self.conv = ng.nn.Sequential(
      ng.nn.Conv2D((3,3),2,1),
      ng.nn.ReLU()
    )
    self.stack = ng.nn.Sequential(
        ng.nn.Linear(100,100),
        ng.nn.ReLU(),
        ng.nn.Linear(100,10),
        ng.nn.Softmax(axis=1)
    )
  
  def forward(self, inputs):
    conv_outputs = self.conv(inputs)
    conv_outputs_flattened = conv_outputs.reshape((inputs.shape[0], 100))
    return self.stack(conv_outputs_flattened)

model = NN()

loss_fn = CE()
optim = Adam(model.get_params(), 0.05)

for iter in range(num_iter):
  optim.zero_grad()
  outputs = model(X_train)
  loss = loss_fn(outputs, y_train)
  loss.backward()
  optim.step()
  print(f"iter {iter+1}/{num_iter}\nloss: {loss}\n")