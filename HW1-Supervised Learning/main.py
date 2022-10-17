import sklearn
from sklearn import svm,neighbors,preprocessing
from sklearn.model_selection import train_test_split,learning_curve,ShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import zero_one_loss
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd
import numpy as np
import seaborn as sns
import time

# record the time
start = time.time()
print("start the code")

# dataset 1
# filename = 'Algerian_forest_fires_dataset_clean.csv' # dataset 1
# Attributes = ["Temperature","RH","Ws","Rain"]
# Target_names = ["not fire","fire"]
# Label_feature_name=["class_label_index"]

# dataset 2
filename = 'Occupancy_Estimation_clean.csv' # dataset 2
Attributes = ["Average Temp","Average Light","Average Sound","CO2_Slope"]
Target_names = ['0','1','2','3']
Target_names1 = ['Occupied','Not Occupied']
Label_feature_name=['Room_Occupancy_Count']

def getdataX(data,Attributes:list):
    X = data.loc[:,Attributes].to_numpy()
    return X
def getdataY(data,Label_feature_name):
    Y = data.loc[:,Label_feature_name].values.tolist()
    Y = np.reshape(Y, (len(Y),))
    return Y

# speed test based on 2 features
def speedtest(Dataset):
    X = Dataset['data'][:, [0,1]]
    y = Dataset['target']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, shuffle=False
    )
    # decision tree
    time0 = time.time()
    print("start the code for decision tree")
    clf = sklearn.tree.DecisionTreeClassifier()
    clf = clf.fit(X_train, y_train)
    time1 = time.time()
    print(['Fitting time for decision tree: ', time1 - time0])
    clf.predict(X_test)
    time2 = time.time()
    print(['Testing time for decision tree: ',time2 - time1])

    # Neural networks
    time0 = time.time()
    print("start the code for Neural Networks")
    clf = MLPClassifier()
    clf = clf.fit(X_train, y_train)
    time1 = time.time()
    print(['Fitting time for neural networks: ', time1 - time0])
    clf.predict(X_test)
    time2 = time.time()
    print(['Testing time for neural networks: ', time2 - time1])

    # Ada Boosting
    time0 = time.time()
    print("start the code for Ada Boost")
    clf = AdaBoostClassifier(sklearn.tree.DecisionTreeClassifier(max_depth=3))
    clf = clf.fit(X_train, y_train)
    time1 = time.time()
    print(['Fitting time for Ada Boost: ', time1 - time0])
    clf.predict(X_test)
    time2 = time.time()
    print(['Testing time for Ada Boost: ', time2 - time1])

    # Support Vector Machines
    time0 = time.time()
    print("start the code for SVM")
    clf = svm.SVC()
    clf = clf.fit(X_train, y_train)
    time1 = time.time()
    print(['Fitting time for SVM: ', time1 - time0])
    clf.predict(X_test)
    time2 = time.time()
    print(['Testing time for SVM: ', time2 - time1])

    # k-nearest neighbors
    time0 = time.time()
    print("start the code for KNN")
    clf = neighbors.KNeighborsClassifier()
    clf = clf.fit(X_train, y_train)
    time1 = time.time()
    print(['Fitting time for KNN: ', time1 - time0])
    clf.predict(X_test)
    time2 = time.time()
    print(['Testing time for KNN: ', time2 - time1])

# Decision trees with some form of pruning
def dtree(Dataset):
    # Parameters
    n_classes = 2
    plot_colors = "ryb"
    plot_step = 0.02

    for pairidx, pair in enumerate([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]):
        # We only take the two corresponding features
        X = Dataset['data'][:, pair]
        y = Dataset['target']

        # Train
        clf = sklearn.tree.DecisionTreeClassifier().fit(X, y)

        # Plot the decision boundary
        ax = plt.subplot(2, 3, pairidx + 1)
        plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)
        DecisionBoundaryDisplay.from_estimator(
            clf,
            X,
            cmap=plt.cm.RdYlBu,
            response_method="predict",
            ax=ax,
            xlabel=Dataset['feature_names'][pair[0]],
            ylabel=Dataset['feature_names'][pair[1]],
        )

        # Plot the training points
        for i, color in zip(range(n_classes), plot_colors):
            idx = np.where(y == i)
            plt.scatter(
                X[idx, 0],
                X[idx, 1],
                c=color,
                label=Dataset['target_names'][i],
                cmap=plt.cm.RdYlBu,
                edgecolor="black",
                s=15,
            )

    plt.suptitle("Decision surface of decision trees trained on pairs of features")
    plt.legend(loc="lower right", borderpad=0, handletextpad=0)
    _ = plt.axis("tight")
    return plt.show()

def dtreewithpruning(Dataset):
    X = Dataset['data']
    y = Dataset['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    clf = sklearn.tree.DecisionTreeClassifier(random_state=0)
    path = clf.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas, impurities = path.ccp_alphas, path.impurities

    fig, ax = plt.subplots()
    ax.plot(ccp_alphas[:-1], impurities[:-1], marker="o", drawstyle="steps-post")
    ax.set_xlabel("effective alpha")
    ax.set_ylabel("total impurity of leaves")
    ax.set_title("Total Impurity vs effective alpha for training set")

    clfs = []
    for ccp_alpha in ccp_alphas:
        clf = sklearn.tree.DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
        clf.fit(X_train, y_train)
        clfs.append(clf)
    print(
        "Number of nodes in the last tree is: {} with ccp_alpha: {}".format(
            clfs[-1].tree_.node_count, ccp_alphas[-1]
        )
    )

    clfs = clfs[:-1]
    ccp_alphas = ccp_alphas[:-1]

    node_counts = [clf.tree_.node_count for clf in clfs]
    depth = [clf.tree_.max_depth for clf in clfs]
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(ccp_alphas, node_counts, marker="o", drawstyle="steps-post")
    ax[0].set_xlabel("alpha")
    ax[0].set_ylabel("number of nodes")
    ax[0].set_title("Number of nodes vs alpha")
    ax[1].plot(ccp_alphas, depth, marker="o", drawstyle="steps-post")
    ax[1].set_xlabel("alpha")
    ax[1].set_ylabel("depth of tree")
    ax[1].set_title("Depth vs alpha")
    fig.tight_layout()

    train_scores = [clf.score(X_train, y_train) for clf in clfs]
    test_scores = [clf.score(X_test, y_test) for clf in clfs]

    fig, ax = plt.subplots()
    ax.set_xlabel("alpha")
    ax.set_ylabel("accuracy")
    ax.set_title("Accuracy vs alpha for training and testing sets")
    ax.plot(ccp_alphas, train_scores, marker="o", label="train", drawstyle="steps-post")
    ax.plot(ccp_alphas, test_scores, marker="o", label="test", drawstyle="steps-post")
    ax.legend()
    plt.show()

def treeplot(Dataset):
    # plot tree
    plt.figure(figsize=(20, 14))
    clf = sklearn.tree.DecisionTreeClassifier().fit(Dataset['data'], Dataset['target'])
    sklearn.tree.plot_tree(clf, filled=True)
    plt.title("Decision tree trained on all features")
    plt.show()

# Neural networks
def neural_networks(Dataset):
    figure = plt.figure(figsize=(17, 9))
    i = 1
    for pairidx, pair in enumerate([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]):
        # We only take the two corresponding features
        X = Dataset['data'][:, pair]
        y = Dataset['target']
        scaler = sklearn.preprocessing.StandardScaler()
        # Don't cheat - fit only on training data
        scaler.fit(X)
        X = scaler.transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25)

        h = 0.02  # step size in the mesh

        alphas = np.logspace(-1, 1, 5)

        classifiers = []
        names = []
        for alpha in alphas:
            classifiers.append(
                make_pipeline(
                    scaler,
                    MLPClassifier(
                        solver="lbfgs",
                        alpha=alpha,
                        random_state=1,
                        max_iter=2000,
                        early_stopping=True,
                        hidden_layer_sizes=[5, 5],
                    ),
                )
            )
            names.append(f"alpha {alpha:.2f}")
        rng = np.random.RandomState(2)
        linearly_separable = (X, y)

        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        # just plot the dataset first
        cm = plt.cm.RdBu
        cm_bright = ListedColormap(["#FF0000", "#0000FF"])
        ax = plt.subplot(6, len(classifiers) + 1, i)
        # Plot the training points
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
        # and testing points
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6)
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        i += 1

        # iterate over classifiers
        for name, clf in zip(names, classifiers):
            ax = plt.subplot(6, len(classifiers) + 1, i)
            clf.fit(X_train, y_train)
            score = clf.score(X_test, y_test)
            # Plot the decision boundary. For that, we will assign a color to each
            # point in the mesh [x_min, x_max] x [y_min, y_max].
            if hasattr(clf, "decision_function"):
                Z = clf.decision_function(np.column_stack([xx.ravel(), yy.ravel()]))
            else:
                Z = clf.predict_proba(np.column_stack([xx.ravel(), yy.ravel()]))[:, 1]

            # Put the result into a color plot
            Z = Z.reshape(xx.shape)
            ax.contourf(xx, yy, Z, cmap=cm, alpha=0.8)

            # Plot also the training points
            ax.scatter(
                X_train[:, 0],
                X_train[:, 1],
                c=y_train,
                cmap=cm_bright,
                edgecolors="black",
                s=25,
            )
            # and testing points
            ax.scatter(
                X_test[:, 0],
                X_test[:, 1],
                c=y_test,
                cmap=cm_bright,
                alpha=0.6,
                edgecolors="black",
                s=25,
            )

            ax.set_xlim(xx.min(), xx.max())
            ax.set_ylim(yy.min(), yy.max())
            ax.set_xticks(())
            ax.set_yticks(())
            ax.set_title(name)
            ax.text(
                xx.max() - 0.3,
                yy.min() + 0.3,
                f"{score:.3f}".lstrip("0"),
                size=15,
                horizontalalignment="right",
            )
            i += 1
    figure.subplots_adjust(left=0.02, right=0.98)
    plt.show()

def nn_loss(Dataset):
    X = Dataset['data']
    y = Dataset['target']
    scaler = preprocessing.StandardScaler()
    scaler.fit(X)
    X_std = scaler.transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_std, y, random_state=0)

    mlp = MLPClassifier(
        activation="relu",
        max_iter=2000,
        validation_fraction=0.2,
        early_stopping=True,
    )
    mlp.fit(X_train, y_train)
    print(mlp.score(X_train, y_train))
    plt.plot(mlp.loss_curve_,label='loss')
    plt.plot(mlp.validation_scores_,label='accuracy')
    plt.xlabel("n iteration")
    plt.legend(loc='upper left')
    plt.show()

# Ada Boosting
def adaboosting(Dataset):
    n_classes = 3
    n_estimators = 30
    cmap = plt.cm.RdYlBu
    plot_step = 0.02  # fine step width for decision surface contours
    plot_step_coarser = 0.5  # step widths for coarse classifier guesses
    RANDOM_SEED = 13  # fix the seed on each iteration
    plot_idx = 1

    models = [
        AdaBoostClassifier(sklearn.tree.DecisionTreeClassifier(max_depth=3), n_estimators=n_estimators),
    ]

    for pair in ([0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]):
        for model in models:
            # We only take the two corresponding features
            X = Dataset['data'][:, pair]
            y = Dataset['target']

            # Shuffle
            idx = np.arange(X.shape[0])
            np.random.seed(RANDOM_SEED)
            np.random.shuffle(idx)
            X = X[idx]
            y = y[idx]

            # Standardize
            # mean = X.mean(axis=0)
            # std = X.std(axis=0)
            # X = (X - mean) / std
            scaler = sklearn.preprocessing.StandardScaler()
            scaler.fit(X)
            X = scaler.transform(X)

            # Train
            model.fit(X, y)

            scores = model.score(X, y)
            # Create a title for each column and the console by using str() and
            # slicing away useless parts of the string
            model_title = str(type(model)).split(".")[-1][:-2][: -len("Classifier")]

            model_details = model_title
            if hasattr(model, "estimators_"):
                model_details += " with {} estimators".format(len(model.estimators_))
            print(model_details + " with features", pair, "has a score of", scores)

            plt.subplot(2, 3, plot_idx)
            # if plot_idx <= len(models):
                # Add a title at the top of each column
                # plt.title(model_title, fontsize=9)

            # Now plot the decision boundary using a fine mesh as input to a
            # filled contour plot
            x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
            y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
            xx, yy = np.meshgrid(
                np.arange(x_min, x_max, plot_step), np.arange(y_min, y_max, plot_step)
            )

            # Plot either a single DecisionTreeClassifier or alpha blend the
            # decision surfaces of the ensemble of classifiers
            if isinstance(model, sklearn.tree.DecisionTreeClassifier):
                Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
                Z = Z.reshape(xx.shape)
                cs = plt.contourf(xx, yy, Z, cmap=cmap)
            else:
                # Choose alpha blend level with respect to the number
                # of estimators
                # that are in use (noting that AdaBoost can use fewer estimators
                # than its maximum if it achieves a good enough fit early on)
                estimator_alpha = 1.0 / len(model.estimators_)
                for tree in model.estimators_:
                    Z = tree.predict(np.c_[xx.ravel(), yy.ravel()])
                    Z = Z.reshape(xx.shape)
                    cs = plt.contourf(xx, yy, Z, alpha=estimator_alpha, cmap=cmap)

            # Build a coarser grid to plot a set of ensemble classifications
            # to show how these are different to what we see in the decision
            # surfaces. These points are regularly space and do not have a
            # black outline
            xx_coarser, yy_coarser = np.meshgrid(
                np.arange(x_min, x_max, plot_step_coarser),
                np.arange(y_min, y_max, plot_step_coarser),
            )
            Z_points_coarser = model.predict(
                np.c_[xx_coarser.ravel(), yy_coarser.ravel()]
            ).reshape(xx_coarser.shape)
            cs_points = plt.scatter(
                xx_coarser,
                yy_coarser,
                s=15,
                c=Z_points_coarser,
                cmap=cmap,
                edgecolors="none",
            )

            # Plot the training points, these are clustered together and have a
            # black outline
            plt.scatter(
                X[:, 0],
                X[:, 1],
                c=y,
                cmap=ListedColormap(["r", "y", "b"]),
                edgecolor="k",
                s=20,
            )
            plot_idx += 1  # move on to the next plot in sequence

    plt.suptitle("AdaBoost - Classifiers on feature subsets of forest fire dataset", fontsize=12)
    plt.axis("tight")
    plt.tight_layout(h_pad=0.2, w_pad=0.2, pad=2.5)
    plt.show()

def adaboosting_evaluation(Dataset):
# Preparing the data and baseline models
    n_estimators = 400
    learning_rate = 1.0

    # X = Dataset['data'][:, :2]
    X = Dataset['data']
    y = Dataset['target']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, shuffle=False
    )
    dt_stump = sklearn.tree.DecisionTreeClassifier(max_depth=1, min_samples_leaf=1)
    dt_stump.fit(X_train, y_train)
    dt_stump_err = 1.0 - dt_stump.score(X_test, y_test)

    dt = sklearn.tree.DecisionTreeClassifier(max_depth=9, min_samples_leaf=1)
    dt.fit(X_train, y_train)
    dt_err = 1.0 - dt.score(X_test, y_test)
# Adaboost with discrete SAMME and real SAMME.R
    ada_discrete = AdaBoostClassifier(
        base_estimator=dt_stump,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        algorithm="SAMME",
    )
    ada_discrete.fit(X_train, y_train)

    ada_real = AdaBoostClassifier(
        base_estimator=dt_stump,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        algorithm="SAMME.R",
    )
    ada_real.fit(X_train, y_train)

    ada_discrete_err = np.zeros((n_estimators,))
    for i, y_pred in enumerate(ada_discrete.staged_predict(X_test)):
        ada_discrete_err[i] = zero_one_loss(y_pred, y_test)

    ada_discrete_err_train = np.zeros((n_estimators,))
    for i, y_pred in enumerate(ada_discrete.staged_predict(X_train)):
        ada_discrete_err_train[i] = zero_one_loss(y_pred, y_train)

    ada_real_err = np.zeros((n_estimators,))
    for i, y_pred in enumerate(ada_real.staged_predict(X_test)):
        ada_real_err[i] = zero_one_loss(y_pred, y_test)

    ada_real_err_train = np.zeros((n_estimators,))
    for i, y_pred in enumerate(ada_real.staged_predict(X_train)):
        ada_real_err_train[i] = zero_one_loss(y_pred, y_train)

    # plot the results

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot([1, n_estimators], [dt_stump_err] * 2, "k-", label="Decision Stump Error")
    ax.plot([1, n_estimators], [dt_err] * 2, "k--", label="Decision Tree Error")

    colors = sns.color_palette("colorblind")

    ax.plot(
        np.arange(n_estimators) + 1,
        ada_discrete_err,
        label="Discrete AdaBoost Test Error",
        color=colors[0],
    )
    ax.plot(
        np.arange(n_estimators) + 1,
        ada_discrete_err_train,
        label="Discrete AdaBoost Train Error",
        color=colors[1],
    )
    ax.plot(
        np.arange(n_estimators) + 1,
        ada_real_err,
        label="Real AdaBoost Test Error",
        color=colors[2],
    )
    ax.plot(
        np.arange(n_estimators) + 1,
        ada_real_err_train,
        label="Real AdaBoost Train Error",
        color=colors[4],
    )

    ax.set_ylim((0.0, 0.5))
    ax.set_xlabel("Number of weak learners")
    ax.set_ylabel("error rate")

    leg = ax.legend(loc="upper right", fancybox=True)
    leg.get_frame().set_alpha(0.7)

    plt.show()

# Support Vector Machines
def SVM(Dataset):
    # Take the first two features. We could avoid this by using a two-dim dataset
    for pair in ([0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]):
        X = Dataset['data'][:, pair]
        # X = Dataset['data']
        y = Dataset['target']

        # we create an instance of SVM and fit out data. We do not scale our
        # data since we want to plot the support vectors
        C = 1.0  # SVM regularization parameter
        # bool class
        models = (
            svm.SVC(kernel="linear", C=C),
            svm.LinearSVC(C=C, max_iter=10000),
            svm.SVC(kernel="rbf", gamma=0.7, C=C),
            svm.SVC(kernel="poly", degree=3, gamma="auto", C=C),
        )

        models = (clf.fit(X, y) for clf in models)

        # title for the plots
        titles = (
            "SVC with linear kernel",
            "LinearSVC (linear kernel)",
            "SVC with RBF kernel",
            "SVC with polynomial (degree 3) kernel",
        )

        # Set-up 2x2 grid for plotting.
        fig, sub = plt.subplots(2, 2)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)

        X0, X1 = X[:, 0], X[:, 1]

        for clf, title, ax in zip(models, titles, sub.flatten()):
            disp = DecisionBoundaryDisplay.from_estimator(
                clf,
                X,
                response_method="predict",
                cmap=plt.cm.coolwarm,
                alpha=0.8,
                ax=ax,
                xlabel=Dataset['feature_names'][pair[0]],
                ylabel=Dataset['feature_names'][pair[1]],
            )
            ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors="k")
            ax.set_xticks(())
            ax.set_yticks(())
            ax.set_title(title)

        # plt.suptitle("SVM - Classifiers on feature subsets of forest fire dataset", fontsize=12)
        plt.show()
def MultiSVM(Dataset):
    # Take the first two features. We could avoid this by using a two-dim dataset
    for pair in ([0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]):
        X = Dataset['data'][:, pair]
        # X = Dataset['data']
        y = Dataset['target']

        # we create an instance of SVM and fit out data. We do not scale our
        # data since we want to plot the support vectors
        C = 1.0  # SVM regularization parameter

        # multiclass classification
        models = (
            svm.SVC(kernel="linear", C=C, decision_function_shape='ovo'),
            svm.SVC(kernel="rbf", gamma=0.7, C=C, decision_function_shape='ovo'),
        )
        models = (clf.fit(X, y) for clf in models)

        # title for the plots
        titles = (
            "SVC with linear kernel",
            "SVC with RBF kernel",
        )
        # Set-up 2x2 grid for plotting.
        fig, sub = plt.subplots(1, 2)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)

        X0, X1 = X[:, 0], X[:, 1]

        for clf, title, ax in zip(models, titles, sub.flatten()):
            disp = DecisionBoundaryDisplay.from_estimator(
                clf,
                X,
                response_method="predict",
                cmap=plt.cm.coolwarm,
                alpha=0.8,
                ax=ax,
                xlabel=Dataset['feature_names'][pair[0]],
                ylabel=Dataset['feature_names'][pair[1]],
            )
            ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors="k")
            ax.set_xticks(())
            ax.set_yticks(())
            ax.set_title(title)

        # plt.suptitle("SVM - Classifiers on feature subsets of forest fire dataset", fontsize=12)
        plt.show()
# k-nearest neighbors
def KNN(Dataset):
    n_neighbors = 5
    plot_colors = "ryb"
    # we only take the first two features. We could avoid this ugly
    # slicing by using a two-dim dataset
    for pair in ([0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]):
        X = Dataset['data'][:, pair]
        y = Dataset['target']

        # Create color maps
        cmap_light = ListedColormap(["orange", "cyan", "cornflowerblue"])
        cmap_bold = ["darkorange", "c", "darkblue"]
        plot_idx=0
        for weights in ["uniform", "distance"]:
            # we create an instance of Neighbours Classifier and fit the data.
            clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
            clf.fit(X, y)

            # _, ax = plt.subplots()
            ax = plt.subplot(2, 1, plot_idx + 1)
            plt.subplots_adjust(wspace=0.4, hspace=0.5)
            DecisionBoundaryDisplay.from_estimator(
                clf,
                X,
                cmap=cmap_light,
                ax=ax,
                response_method="predict",
                plot_method="pcolormesh",
                xlabel=Dataset['feature_names'][pair[0]],
                ylabel=Dataset['feature_names'][pair[1]],
                shading="auto",
            )

            # Plot also the training points
            # sns.scatterplot(
            #     x=X[:, 0],
            #     y=X[:, 1],
            #     # hue=Dataset['target_names'][],
            #     palette=cmap_bold,
            #     alpha=1.0,
            #     edgecolor="black",
            # )
            for i, color in zip(range(len(Dataset['target_names'])), plot_colors):
                idx = np.where(y == i)
                plt.scatter(
                    X[idx, 0],
                    X[idx, 1],
                    c=color,
                    label=Dataset['target_names'][i],
                    cmap=plt.cm.RdYlBu,
                    edgecolor="black",
                    s=15,
                )
            plt.title(
                "2-Class classification (k = %i, weights = '%s')" % (n_neighbors, weights)
            )
            plot_idx += 1

        plt.show()



# plot learning curve
def plot_learning_curve(
    estimator,
    title,
    X,
    y,
    axes=None,
    ylim=None,
    cv=None,
    n_jobs=None,
    scoring=None,
    train_sizes=np.linspace(0.1, 1.0, 5),
):
    """
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.

    Parameters
    ----------
    estimator : estimator instance
        An estimator instance implementing `fit` and `predict` methods which
        will be cloned for each validation.

    title : str
        Title for the chart.

    X : array-like of shape (n_samples, n_features)
        Training vector, where ``n_samples`` is the number of samples and
        ``n_features`` is the number of features.

    y : array-like of shape (n_samples) or (n_samples, n_features)
        Target relative to ``X`` for classification or regression;
        None for unsupervised learning.

    axes : array-like of shape (3,), default=None
        Axes to use for plotting the curves.

    ylim : tuple of shape (2,), default=None
        Defines minimum and maximum y-values plotted, e.g. (ymin, ymax).

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    scoring : str or callable, default=None
        A str (see model evaluation documentation) or
        a scorer callable object / function with signature
        ``scorer(estimator, X, y)``.

    train_sizes : array-like of shape (n_ticks,)
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the ``dtype`` is float, it is regarded
        as a fraction of the maximum size of the training set (that is
        determined by the selected validation method), i.e. it has to be within
        (0, 1]. Otherwise it is interpreted as absolute sizes of the training
        sets. Note that for classification the number of samples usually have
        to be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples percent %")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
        estimator,
        X,
        y,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        train_sizes=train_sizes,
        return_times=True,
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    plt.subplots_adjust(wspace=0.4, hspace=0.45,left=0.2,right=0.8)
    axes[0].grid()
    axes[0].fill_between(
        train_sizes/len(X)*100,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r",
    )
    axes[0].fill_between(
        train_sizes/len(X)*100,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="g",
    )
    axes[0].plot(
        train_sizes/len(X)*100, train_scores_mean, "o-", color="r", label="Training score"
    )
    axes[0].plot(
        train_sizes/len(X)*100, test_scores_mean, "o-", color="g", label="Cross-validation score"
    )
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes/len(X)*100, fit_times_mean, "o-")
    axes[1].fill_between(
        train_sizes/len(X)*100,
        fit_times_mean - fit_times_std,
        fit_times_mean + fit_times_std,
        alpha=0.1,
    )
    axes[1].set_xlabel("Training examples percent %")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    fit_time_argsort = fit_times_mean.argsort()
    fit_time_sorted = fit_times_mean[fit_time_argsort]
    test_scores_mean_sorted = test_scores_mean[fit_time_argsort]
    test_scores_std_sorted = test_scores_std[fit_time_argsort]
    axes[2].grid()
    axes[2].plot(fit_time_sorted, test_scores_mean_sorted, "o-")
    axes[2].fill_between(
        fit_time_sorted,
        test_scores_mean_sorted - test_scores_std_sorted,
        test_scores_mean_sorted + test_scores_std_sorted,
        alpha=0.1,
    )
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    return plt.show()


# Main code
data=pd.read_csv(filename).convert_dtypes()
data_train, data_test = train_test_split(data , test_size=0.25)
# Attributes = ["Temperature","RH"]

X_train = getdataX(data_train,Attributes)
Y_train = getdataY(data_train,Label_feature_name)

X_test = getdataX(data_test,Attributes)
Y_test = getdataY(data_test,Label_feature_name)

X = getdataX(data,Attributes)
Y = getdataY(data,Label_feature_name)

Dataset = {'data':X,'target':Y,'target_names':Target_names,'feature_names':Attributes}
# print(Dataset['data'])

# Tree
dtree(Dataset)
treeplot(Dataset)
dtreewithpruning(Dataset)

# neural networks
neural_networks(Dataset)
nn_loss(Dataset)

# adaboosting
adaboosting(Dataset)

# SVM
# SVM(Dataset)
MultiSVM(Dataset)

# Knn
KNN(Dataset)

speedtest(Dataset)

# CURVES
models = [
    sklearn.tree.DecisionTreeClassifier(max_depth=None),
    MLPClassifier(solver="lbfgs",
                        # alpha=alpha,
                        random_state=1,
                        max_iter=2000,
                        early_stopping=True,
                        hidden_layer_sizes=[5, 5],),
    AdaBoostClassifier(sklearn.tree.DecisionTreeClassifier(max_depth=3), n_estimators=30),
    svm.SVC(),
    neighbors.KNeighborsClassifier(5, weights='uniform'),
]

# plot leaning curve sample
titles=["Learning Curves (Decision Tree)","Learning Curves (Neural networks)",
       "Learning Curves (Ada Boost)","Learning Curves (Support Vector Machines)","Learning Curves (k(5)-nearest neighbors(uniform))"]


for i in range(0,5):
    fig, axes = plt.subplots(3, 1, figsize=(6, 9))
    title = titles[i]
    if i == 1:
        scaler = sklearn.preprocessing.StandardScaler()
        scaler.fit(X)
        X_input = scaler.transform(X)
    else:
        X_input=X

    # Cross validation with 50 iterations to get smoother mean test and train
    # score curves, each time with 20% data randomly selected as a validation set.
    cv = ShuffleSplit(n_splits=50, test_size=0.2, random_state=0)
    estimator = models[i]
    plot_learning_curve(
        estimator,
        title,
        X_input,
        Y,
        axes=axes,
        ylim=(0.5, 1.01),
        cv=cv,
        n_jobs=4,
        scoring="accuracy",
    )


end = time.time()
print("Run time:",end-start)
