from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from collections import OrderedDict

d_tree = OrderedDict()
d_tree['max_features'] = ('cont', (0.1, 0.99))
d_tree['max_depth'] = ('int', (4, 30))
d_tree['min_samples_split'] = ('cont', (0.1, 0.99))


class Tree(object):
    def __init__(self, problem='binary', max_features=0.5, max_depth=1, min_samples_split=2):
        self.problem = problem
        self.max_features = max_features
        self.max_depth = int(max_depth)
        self.min_samples_split = min_samples_split
        self.name = 'Tree'

    def eval(self):
        if self.problem == 'binary':
            mod = DecisionTreeClassifier(max_features=self.max_features, max_depth=self.max_depth,
                                         min_samples_split=self.min_samples_split, random_state=20)
        else:
            mod = DecisionTreeRegressor(max_features=self.max_features, max_depth=self.max_depth,
                                        min_samples_split=self.min_samples_split, random_state=20)
        return mod
