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

    # path = "../result/hyperband_logistic_0"
    # plots.plot_hyperband_only(path, 1, 'logistic_', 'sgd_', 'mnist', 0)

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

    # path1 = "../result/hyperband_logistic_1"
    # path2 = "../result/tpe_logistic_1"
    # path3 = "../result/hoo_logistic_1"
    # path4 = "../result/hct_logistic_1"
    # path5 = "../result/random_logistic_1"
    # path6 = "../result/hyperloop_logistic_1"
    # paths = [path1, path2, path3, path4, path5, path6]
    # # plots.plot_hoo(path3, 1, 'logistic_', 'sgd_', 'mnist', 0)
    # plots.plot_all(paths, 10, 'logistic_', 'sgd_', 'mnist', 1, 'epochs', type_plot='linear', devs=False)

    path1 = "../result/hyperband_mlp_0"
    path2 = "../result/tpe_mlp_0"
    path3 = "../result/hoo_mlp_0"
    path4 = "../result/hct_mlp_0"
    path5 = "../result/random_mlp_0"
    path6 = "../result/hyperloop_mlp_0"
    paths = [path1, path2, path3, path4, path5, path6]
    # plots.plot_hoo(path3, 1, 'mlp_', 'sgd_', 'mnist', 0)
    plots.plot_all(paths, 8, 'mlp_', 'sgd_', 'mnist_bis', 0, 'epochs', type_plot='linear', devs=False)

    # names = ['ada_', 'gbm_', 'knn_', 'rf_', 'sk_mlp_', 'svm_', 'tree_']
    # for name in names:
    #     path = "../result/hyperband_" + name + "0"
    #     plots.plot_hyperband_only(path, 1, name, '', 'wine', 0)

    # path = "../result/hct_ada_0"
    # plots.plot_hct(path, 1, 'ada_', '', 'wine', 0)

    # names = ['ada_', 'gbm_', 'knn_', 'sk_mlp_', 'svm_']
    # for name in names:
    #     path0 = "../result/hyperband_" + name + "1"
    #     path1 = "../result/tpe_" + name + "1"
    #     path2 = "../result/hoo_" + name + "1"
    #     path3 = "../result/hct_" + name + "1"
    #     path4 = "../result/random_" + name + "1"
    #     path5 = "../result/hyperloop_" + name + "1"
    #     paths = [path0, path1, path2, path3, path4, path5]
    #     plots.plot_all(paths, 20, name, '', 'breast_cancer', 1, 'iterations', type_plot='linear', devs=False)

    # path = "../result/hyperloop_mlp_1/"
    # plots.plot_hyperloop_only(path, 4, 'mlp_', 'sgd_', 'mnist', 1)
