import _setup
import neograd as ng
import numpy as np
from neograd.nn.loss import CE
from neograd.nn.optim import Adam
from neograd.nn.utils import grad_check
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y)

num_train, num_test = 10, 10
num_iter = 1000

def one_hot(cls_arr, num_examples, num_classes):
  encoded = np.zeros((num_examples, num_classes))
  for i in range(num_examples):
    for j in range(num_classes):
      if j==cls_arr[i]:
        encoded[i][j] = 1
  return encoded

X_train_norm = (X_train - np.mean(X_train, keepdims=True))/np.std(X_train, keepdims=True)
X_test_norm = (X_test - np.mean(X_test, keepdims=True))/np.std(X_test, keepdims=True)

# X_train = ng.tensor(X_train_norm[:num_train,:].reshape(num_train,8,8))
# X_test = ng.tensor(X_test_norm[:num_test,:].reshape(num_test,8,8))
# y_train = ng.tensor(one_hot(y_train[:num_train], num_train, 10))
# y_test = ng.tensor(one_hot(y_test[:num_test], num_test, 10))

X_train = ng.tensor(X_train_norm[:num_train,:])
X_test = ng.tensor(X_test_norm[:num_test,:])
y_train = ng.tensor(one_hot(y_train[:num_train], num_train, 10))
y_test = ng.tensor(one_hot(y_test[:num_test], num_test, 10))

# print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

class NN(ng.nn.Model):
  def __init__(self):
    super().__init__(self)
    self.stack = ng.nn.Sequential(
      ng.nn.Linear(64,100),
      ng.nn.ReLU(),
      ng.nn.Linear(100,50),
      ng.nn.Tanh(),
      ng.nn.Linear(50,25),
      ng.nn.ReLU(),
      ng.nn.Linear(25,10),
      ng.nn.Softmax(1)
    )
  
  def forward(self, inputs):
    return self.stack(inputs)

model = NN()

loss_fn = CE()
optim = Adam(model.get_params(), 5e-3)

# for iter in range(num_iter):
#   optim.zero_grad()
#   outputs = model(X_train)
#   loss = loss_fn(outputs, y_train)
#   loss.backward()
#   optim.step()
#   # print(f"iter {iter+1}/{num_iter}\nloss: {loss}\n")
#   if iter%50==0:
#     print(f"iter {iter+1}/{num_iter}\nloss: {loss}\n")

grad_check(model, X_train, y_train, loss_fn)