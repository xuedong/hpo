from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from collections import OrderedDict

d_rf = OrderedDict()
d_rf['n_estimators'] = ('int', (10, 50))
d_rf['min_samples_split'] = ('cont', (0.1, 0.5))
d_rf['max_features'] = ('cont', (0.1, 0.5))


class RF(object):
    def __init__(self, problem='binary', n_estimators=10, max_features=0.5,
                 min_samples_split=0.3, min_samples_leaf=0.2):
        self.problem = problem
        self.n_estimators = int(n_estimators)
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.name = 'RF'

    def eval(self):
        if self.problem == 'binary':
            mod = RandomForestClassifier(n_estimators=self.n_estimators,
                                         max_features=self.max_features,
                                         min_samples_split=self.min_samples_split,
                                         min_samples_leaf=self.min_samples_leaf,
                                         n_jobs=-1,
                                         random_state=20)
        else:
            mod = RandomForestRegressor(n_estimators=self.n_estimators,
                                        max_features=self.max_features,
                                        min_samples_split=self.min_samples_split,
                                        min_samples_leaf=self.min_samples_leaf,
                                        n_jobs=-1,
                                        random_state=20)
        return mod
