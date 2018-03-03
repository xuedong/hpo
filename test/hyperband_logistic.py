import plots


path = "../result/"
plots.plot_hyperband(path, 2, 10, 'logistic_sgd', 'mnist')
