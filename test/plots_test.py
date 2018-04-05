import source.plots as plots


if __name__ == "__main__":
    # path = "../result/hyperband_logistic_0/"
    # plots.plot_hyperband(path, 3, 10, 'logistic_', 'sgd_', 'mnist', 0)
    # path = "../result/hyperband_mlp_0/"
    # plots.plot_hyperband(path, 3, 5, 'mlp_', 'sgd_', 'mnist', 0)
    # path = "../result/hyperband_logistic_2"
    # plots.plot_hyperband(path, 0, 5, 'logistic_', 'sgd_', 'mnist', 2)

    # path = "../result/random_logistic_0"
    # plots.plot_random(path, 1, 'logistic_', 'sgd_', 'mnist', 0)

    # path = "../result/hyperband_logistic_2"
    # plots.plot_hyperband_only(path, 5, 'logistic_', 'sgd_', 'mnist', 2)

    # path = "../result/tpe_logistic_0/"
    # plots.plot_tpe(path, 10, 'logistic_', 'sgd_', 'mnist', 0)
    # path = "../result/tpe_mlp_0"
    # plots.plot_tpe(path, 5, 'mlp_', 'sgd_', 'mnist', 0)

    # path1 = "../result/hyperband_mlp_0"
    # path2 = "../tpe_mlp_0"
    # paths = [path1, path2]
    # plots.plot_all(paths, 3, 5, 'mlp_', 'sgd_', 'mnist', 0)

    # path1 = "../result/hyperband_logistic_0"
    # path2 = "../tpe_logistic_0"
    # paths = [path1, path2]
    # plots.plot_all(paths, 3, 10, 'logistic_', 'sgd_', 'mnist', 0, devs=False)

    # path1 = "../result/hyperband_logistic_0"
    # path2 = "../tpe_logistic_0"
    # path3 = "../hoo_logistic_0"
    # path4 = "../hct_logistic_0"
    # path5 = "../random_logistic_0"
    # paths = [path1, path2, path3, path4, path5]
    # # plots.plot_ho(path, 10, 'logistic_', 'sgd_', 'mnist', 0)
    # plots.plot_all(paths, 3, 'logistic_', 'sgd_', 'mnist', 0, devs=False)

    path = "../result/hyperband_ada_0"
    plots.plot_hyperband_only(path, 1, 'ada_', '', 'wine', 0)
