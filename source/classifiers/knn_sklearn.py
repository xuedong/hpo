from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from collections import OrderedDict

d_knn = OrderedDict()
d_knn['n_neighbors'] = ('int', (10, 50))


class KNN(object):
    def __init__(self, problem='binary', n_neighbors=5, leaf_size=30):
        self.problem = problem
        self.n_neighbors = int(n_neighbors)
        self.leaf_size = int(leaf_size)
        self.name = 'KNN'

    def eval(self):
        if self.problem == 'binary':
            mod = KNeighborsClassifier(n_neighbors=self.n_neighbors,
                                       leaf_size=self.leaf_size)
        else:
            mod = KNeighborsRegressor(n_neighbors=self.n_neighbors,
                                      leaf_size=self.leaf_size)
        return mod
