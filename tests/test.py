import _setup
import neograd as ng
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split

X, y = make_circles(n_samples=1000, noise=0.05, random_state=100)
X_train, X_test, y_train, y_test = train_test_split(X,y)

X_train, X_test = ng.tensor(X_train.T), ng.tensor(X_test.T)
y_train, y_test = ng.tensor(y_train.T.reshape(1,750)), ng.tensor(y_test.T.reshape(1,250))




