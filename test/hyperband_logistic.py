import plots


path = "../result/"
plots.plot_hyperband(path, 4, 10, 'logistic_sgd', 'mnist')
