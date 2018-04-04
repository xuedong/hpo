from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from collections import OrderedDict

d_gbm = OrderedDict()
d_gbm['learning_rate'] = ('cont', (10e-5, 1e-1))
d_gbm['n_estimators'] = ('int', (10, 100))
d_gbm['max_depth'] = ('int', (2, 100))
d_gbm['min_samples_split'] = ('int', (2, 100))


class GBM(object):
    def __init__(self, problem='binary', learning_rate=0.1, n_estimators=100, max_depth=3, min_samples_split=2,
                 min_samples_leaf=1, min_weight_fraction_leaf=0.0, subsample=1.0, max_features=1.0):
        self.problem = problem
        self.learning_rate = learning_rate
        self.n_estimators = int(n_estimators)
        self.max_depth = int(max_depth)
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.subsample = subsample
        self.max_features = max_features
        self.name = 'GBM'

    def eval(self):
        if self.problem == 'binary':
            mod = GradientBoostingClassifier(learning_rate=self.learning_rate, n_estimators=self.n_estimators,
                                             max_depth=self.max_depth, min_samples_split=self.min_samples_split,
                                             min_samples_leaf=self.min_samples_leaf,
                                             min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                                             subsample=self.subsample,
                                             max_features=self.max_features,
                                             random_state=20)
        else:
            mod = GradientBoostingRegressor(learning_rate=self.learning_rate, n_estimators=self.n_estimators,
                                            max_depth=self.max_depth, min_samples_split=self.min_samples_split,
                                            min_samples_leaf=self.min_samples_leaf,
                                            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                                            subsample=self.subsample,
                                            max_features=self.max_features,
                                            random_state=20)
        return mod
