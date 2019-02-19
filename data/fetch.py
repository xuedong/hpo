from sklearn.datasets import fetch_openml
from six.moves import cPickle

if __name__ == '__main__':
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
    with open('mnist_openml.pkl', 'wb') as file:
        cPickle.dump([X, y], file)
