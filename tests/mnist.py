import _setup
import neograd as ng
import numpy as np
from neograd.nn.loss import SoftmaxCE
from neograd.nn.optim import Adam
from neograd.autograd.utils import grad_check
from neograd.nn.utils import get_batches
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y)

num_train, num_test = X_train.shape[0], X_test.shape[0]
# num_train, num_test = 50,50
num_iter = 200

X_train_norm = (X_train - np.mean(X_train, keepdims=True))/np.std(X_train, keepdims=True)
X_test_norm = (X_test - np.mean(X_test, keepdims=True))/np.std(X_test, keepdims=True)

X_train = ng.tensor(X_train_norm[:num_train,:].reshape(num_train,8,8))
X_test = ng.tensor(X_test_norm[:num_test,:].reshape(num_test,8,8))
y_train = ng.tensor(np.eye(10)[y_train[:num_train]])
y_test = ng.tensor(y_test[:num_test])

class NN(ng.nn.Model):
  def __init__(self):
    self.conv = ng.nn.Sequential(
      ng.nn.Conv2D((3,3)),
      ng.nn.ReLU()
    )
    self.stack = ng.nn.Sequential(
        ng.nn.Linear(36,10),
        ng.nn.Tanh()
    )

  def forward(self, inputs):
    conv_outputs = self.conv(inputs)
    conv_outputs_flattened = conv_outputs.reshape((inputs.shape[0], 36))
    return self.stack(conv_outputs_flattened)

CHKPT_PATH = '/Users/pranavsastry/Downloads/mnist_checkpoints' 
WGTS_PATH = '/Users/pranavsastry/Downloads/mnist_weights.hkl'

model = NN()
loss_fn = SoftmaxCE(axis=1)
optim = Adam(model.get_params(), 5e-3)

# chkpt = ng.Checkpoint(model, CHKPT_PATH)
batch_size = 200

for iter in range(num_iter):
  for batch_input, batch_target in get_batches(X_train, y_train, batch_size):
    optim.zero_grad()
    outputs = model(batch_input)
    loss = loss_fn(outputs, batch_target)
    loss.backward()
    optim.step()
  if iter%50==0:
    print(f"iter {iter+1}/{num_iter}\nloss: {loss}\n")
    # chkpt.add(
    #   iter = iter,
    #   loss = float(loss.data)
    # )

with model.eval():
  test_outputs = model(X_test)
  probs = ng.nn.Softmax(1)(test_outputs.data)
  preds = np.argmax(probs.data, axis=1)

report = classification_report(y_test.data.astype(int).flatten(), preds.flatten())
print(report)
accuracy = accuracy_score(y_test.data.astype(int).flatten(), preds.flatten())
print('Accuracy:', accuracy)

grad_check(model, X_train, y_train, loss_fn)