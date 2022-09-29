import _setup
import neograd as ng
import numpy as np
from neograd.nn.loss import CE
from neograd.nn.optim import Adam
from neograd.autograd.utils import grad_check
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y)

num_train, num_test = X_train.shape[0], X_test.shape[0]
num_iter = 500

def one_hot(cls_arr, num_examples, num_classes):
  encoded = np.zeros((num_examples, num_classes))
  for i in range(num_examples):
    for j in range(num_classes):
      if j==cls_arr[i]:
        encoded[i][j] = 1
  return encoded

X_train_norm = (X_train - np.mean(X_train, keepdims=True))/np.std(X_train, keepdims=True)
X_test_norm = (X_test - np.mean(X_test, keepdims=True))/np.std(X_test, keepdims=True)

X_train = ng.tensor(X_train_norm[:num_train,:].reshape(num_train,8,8))
X_test = ng.tensor(X_test_norm[:num_test,:].reshape(num_test,8,8))
y_train = ng.tensor(one_hot(y_train[:num_train], num_train, 10))
y_test = ng.tensor(y_test[:num_test])

class NN(ng.nn.Model):
  def __init__(self):
    self.conv = ng.nn.Sequential(
      ng.nn.Conv2D((3,3)),
      ng.nn.ReLU()
    )
    self.stack = ng.nn.Sequential(
        ng.nn.Linear(36,10),
        ng.nn.Softmax(1)
    )

  def forward(self, inputs):
    conv_outputs = self.conv(inputs)
    conv_outputs_flattened = conv_outputs.reshape((inputs.shape[0], 36))
    return self.stack(conv_outputs_flattened)

model = NN()

loss_fn = CE()
optim = Adam(model.get_params(), 5e-3)

for iter in range(num_iter):
  optim.zero_grad()
  outputs = model(X_train)
  loss = loss_fn(outputs, y_train)
  loss.backward()
  optim.step()
  if iter%50==0:
    print(f"iter {iter+1}/{num_iter}\nloss: {loss}\n")

with model.eval():
  test_outputs = model(X_test)
  preds = np.argmax(test_outputs.data, axis=1)

print(classification_report(y_test.data.astype(int).flatten(), preds.flatten()))
print(accuracy_score(y_test.data.astype(int).flatten(), preds.flatten()))

# grad_check(model, X_train, y_train, loss_fn)