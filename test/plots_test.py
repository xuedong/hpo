import src.plots as plots


if __name__ == "__main__":
    # path = "../result/hyperband_logistic_0/"
    # plots.plot_hyperband(path, 3, 10, 'logistic_', 'sgd_', 'mnist', 0)
    # path = "../result/hyperband_mlp_0/"
    # plots.plot_hyperband(path, 3, 5, 'mlp_', 'sgd_', 'mnist', 0)

    # path = "../result/tpe_logistic_0/"
    # plots.plot_tpe(path, 10, 'logistic_', 'sgd_', 'mnist', 0)
    # path = "../result/tpe_mlp_0"
    # plots.plot_tpe(path, 5, 'mlp_', 'sgd_', 'mnist', 0)

    path1 = "../result/hyperband_mlp_0"
    path2 = "../tpe_mlp_0"
    plots.plot_all(path1, path2, 3, 5, 'mlp_', 'sgd_', 'mnist', 0, devs=True)

    # path1 = "../result/hyperband_logistic_0"
    # path2 = "../tpe_logistic_0"
    # plots.plot_all(path1, path2, 3, 10, 'logistic_', 'sgd_', 'mnist', 0, devs=False)
