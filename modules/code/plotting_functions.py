import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection import cross_validate, train_test_split

def make_bracket(s, xy, textxy, width, ax):
    annotation = ax.annotate(
        s, xy, textxy, ha="center", va="center", size=20,
        arrowprops=dict(arrowstyle="-[", fc="w", ec="k",
                        lw=2,), bbox=dict(boxstyle="square", fc="w"))
    annotation.arrow_patch.get_arrowstyle().widthB = width

def plot_improper_processing(estimator_name, transformer_name = 'CountVectorizer'):
    # Adapted from https://github.com/amueller/mglearn/blob/106cf48ef03710ef1402813997746741aa6467da/mglearn/plot_improper_preprocessing.py#L12
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))

    for axis in axes:
        bars = axis.barh([0, 0, 0], [11.9, 2.9, 4.9], left=[0, 12, 15],
                         color=['white', 'grey', 'grey'], hatch="//",
                         align='edge', edgecolor='k')
        bars[2].set_hatch(r"")
        axis.set_yticks(())
        axis.set_frame_on(False)
        axis.set_ylim(-.1, 6)
        axis.set_xlim(-0.1, 20.1)
        axis.set_xticks(())
        axis.tick_params(length=0, labeltop=True, labelbottom=False)
        axis.text(6, -.3, "training folds",
                  fontdict={'fontsize': 14}, horizontalalignment="center")
        axis.text(13.5, -.3, "validation fold",
                  fontdict={'fontsize': 14}, horizontalalignment="center")
        axis.text(17.5, -.3, "test set",
                  fontdict={'fontsize': 14}, horizontalalignment="center")

    make_bracket(transformer_name + " fit", (7.5, 1.3), (7.5, 2.), 15, axes[0])
    make_bracket(estimator_name + " fit", (6, 3), (6, 4), 12, axes[0])
    make_bracket(estimator_name + "predict", (13.4, 3), (13.4, 4), 2.5, axes[0])

    axes[0].set_title("Cross validation")
    axes[1].set_title("Test set prediction")

    make_bracket(transformer_name + " fit", (7.5, 1.3), (7.5, 2.), 15, axes[1])
    make_bracket(estimator_name + " fit", (7.5, 3), (7.5, 4), 15, axes[1])
    make_bracket(estimator_name + " predict", (17.5, 3), (17.5, 4), 4.8, axes[1])    
    
    
def plot_proper_processing(estimator_name, transformer_name='CountVectorizer'):
    # Adapted from https://github.com/amueller/mglearn/blob/106cf48ef03710ef1402813997746741aa6467da/mglearn/plot_improper_preprocessing.py#L12
    
    fig, axes = plt.subplots(2, 1, figsize=(15, 8))

    for axis in axes:
        bars = axis.barh([0, 0, 0], [11.9, 2.9, 4.9],
                         left=[0, 12, 15], color=['white', 'grey', 'grey'],
                         hatch="//", align='edge', edgecolor='k')
        bars[2].set_hatch(r"")
        axis.set_yticks(())
        axis.set_frame_on(False)
        axis.set_ylim(-.1, 4.5)
        axis.set_xlim(-0.1, 20.1)
        axis.set_xticks(())
        axis.tick_params(length=0, labeltop=True, labelbottom=False)
        axis.text(6, -.3, "training folds", fontdict={'fontsize': 14},
                  horizontalalignment="center")
        axis.text(13.5, -.3, "validation fold", fontdict={'fontsize': 14},
                  horizontalalignment="center")
        axis.text(17.5, -.3, "test set", fontdict={'fontsize': 14},
                  horizontalalignment="center")

    make_bracket(transformer_name + "fit", (6, 1.3), (6, 2.), 12, axes[0])
    make_bracket(estimator_name + " fit", (6, 3), (6, 4), 12, axes[0])
    make_bracket(estimator_name + " predict", (13.4, 3), (13.4, 4), 2.5, axes[0])

    axes[0].set_title("Cross validation")
    axes[1].set_title("Test set prediction")

    make_bracket("scaler fit", (7.5, 1.3), (7.5, 2.), 15, axes[1])
    make_bracket(estimator_name + " fit", (7.5, 3), (7.5, 4), 15, axes[1])
    make_bracket(estimator_name + " predict", (17.5, 3), (17.5, 4), 4.8, axes[1])
    fig.subplots_adjust(hspace=.3)
    
def plot_original_scaled(
    X_train,
    X_test,
    train_transformed,
    test_transformed,
    title_transformed="Properly transformed",
):
    fig, axes = plt.subplots(1, 2, figsize=(15, 4))
    axes[0].scatter(X_train[:, 0], X_train[:, 1], label="Training set", s=60)
    axes[0].scatter(
        X_test[:, 0],
        X_test[:, 1],
        marker="^",
        label="Test set",
        s=60,
    )
    axes[0].legend(loc="upper right")

    axes[0].set_title("Original Data")

    axes[1].scatter(
        train_transformed[:, 0], train_transformed[:, 1], label="Training set", s=60
    )
    axes[1].scatter(
        test_transformed[:, 0],
        test_transformed[:, 1],
        marker="^",
        label="Test set",
        s=60,
    )
    axes[1].legend(loc="upper right")
    axes[1].set_title(title_transformed);    
    
   
# Copied from https://github.com/amueller/mglearn/blob/master/mglearn/plot_scaling.py
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.preprocessing import (StandardScaler, MinMaxScaler, Normalizer,
                                   RobustScaler)

def plot_scaling():
    X, y = make_blobs(n_samples=50, centers=2, random_state=4, cluster_std=1)
    X += 3

    plt.figure(figsize=(15, 8))
    main_ax = plt.subplot2grid((2, 4), (0, 0), rowspan=2, colspan=2)

    main_ax.scatter(X[:, 0], X[:, 1], c=y, s=60)
    maxx = np.abs(X[:, 0]).max()
    maxy = np.abs(X[:, 1]).max()

    main_ax.set_xlim(-maxx + 1, maxx + 1)
    main_ax.set_ylim(-maxy + 1, maxy + 1)
    main_ax.set_title("Original Data")
    other_axes = [plt.subplot2grid((2, 4), (i, j))
                  for j in range(2, 4) for i in range(2)]

    for ax, scaler in zip(other_axes, [StandardScaler(), RobustScaler(),
                                       MinMaxScaler(), Normalizer(norm='l2')]):
        X_ = scaler.fit_transform(X)
        ax.scatter(X_[:, 0], X_[:, 1], c=y, s=60)
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_title(type(scaler).__name__)

    other_axes.append(main_ax)

    for ax in other_axes:
        ax.spines['left'].set_position('center')
        ax.spines['right'].set_color('none')
        ax.spines['bottom'].set_position('center')
        ax.spines['top'].set_color('none')
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')    