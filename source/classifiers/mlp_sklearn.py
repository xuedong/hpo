from sklearn.neural_network import MLPClassifier, MLPRegressor
from collections import OrderedDict

d_mlp = OrderedDict()
d_mlp['hidden_layer_size'] = ('int', (5, 50))
d_mlp['alpha'] = ('cont', (1e-5, 0.9))


class MLP(object):
    def __init__(self, problem='binary', hidden_layer_size=100, alpha=10e-4,
                 learning_rate_init=10e-4, beta_1=0.9, beta_2=0.999):
        self.problem = problem
        self.hidden_layer_sizes = (int(hidden_layer_size),)
        self.alpha = alpha
        self.learning_rate_init = learning_rate_init
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.name = 'MLP'

    def eval(self):
        if self.problem == 'binary':
            mod = MLPClassifier(hidden_layer_sizes=self.hidden_layer_sizes, alpha=self.alpha,
                                learning_rate_init=self.learning_rate_init, beta_1=self.beta_1, beta_2=self.beta_2,
                                random_state=20)
        else:
            mod = MLPRegressor(hidden_layer_sizes=self.hidden_layer_sizes, alpha=self.alpha,
                               learning_rate_init=self.learning_rate_init, beta_1=self.beta_1, beta_2=self.beta_2,
                               random_state=20)
        return mod
