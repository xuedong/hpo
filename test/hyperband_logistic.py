import plots


path = "../result/hyperband_logistic_0/"
plots.plot_hyperband(path, 3, 10, 'logistic_', 'sgd_', 'mnist', 0)
