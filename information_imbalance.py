from collections import defaultdict
import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm
import matplotlib.pyplot as plt


# =============================================================================
# Information Imbalance related functions
# =============================================================================


def _vectorized_information_imbalance(A, B, metric):
    """
    All-NumPy version of information imbalance calculation.
    """
    N = A.shape[0]
    nn_A = np.argsort(cdist(A, A, metric=metric), axis=1)[:, 1]
    r_B = np.argsort(cdist(B, B, metric=metric), axis=1)
    r_B = [np.where(r == n)[0][0] for r, n in zip(r_B, nn_A)]
    cond_r_B = np.sum(r_B)
    imbalance = 2 * cond_r_B / N**2
    return imbalance


def _sequential_information_imbalance(A, B, metric):
    """
    Hybrid Python-NumPy version of information imbalance calculation.
    """
    N = A.shape[0]
    cond_r_B = 0.0
    for p in range(N):
        A_p = np.atleast_2d(A[p])
        B_p = np.atleast_2d(B[p])
        nn_A = np.argsort(cdist(A, A_p, metric=metric), axis=0)[1][0]
        r_B = np.argsort(cdist(B, B_p, metric=metric), axis=0)
        r_B = np.where(r_B == nn_A)[0][0]
        cond_r_B += r_B
    imbalance = 2 * cond_r_B / N**2
    return imbalance


def information_imbalance(A, B, metric="euclidean", mode="vectorized"):
    """
    Computes the information imbalance of going from a set of features
    A to a set of features B.

        Delta(A->B) = 2 <r^B | r^A=1> / N

    where r^B is the rank of B and r^A is the rank of A.
    This amounts to computing the rank r_ij^B for each pair i,j for
    which r_ij^A=1.
    The algorithm then scales as N^2.
    Arguments
    ---------
    A        : ndarray, (num_samples, num_features_A) or (num_samples,)
             First set of features.
    B        : ndarray, (num_samples, num_features_B) or (num_samples,)
             Second set of features
    metric   : str
             Metric used to compute the distances. Must be one of the allowed
             metrics in the `cdist` function of SciPy (the one used to actually
             compute distances).
             See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html
    mode     : str
             How the Information Imbalance is computed.
             * 'vectorized' is faster but needs more memory. If it fails,
               it switches to the 'sequential' mode.
             * 'sequential' is slower, but should not run into problems when
               the amount of data is large.
    Returns
    -------
    information_imbalance : float
                          The information imbalance of going from A to B.
    References
    ----------
    Glielmo et al., https://arxiv.org/abs/2104.15079
    """
    # Same number of points
    if A.shape[0] != B.shape[0]:
        raise RuntimeError("Number of points mismatch between A and B.")
    # At least 2D
    if len(A.shape) < 2:
        A = A.reshape(-1, 1)
    if len(B.shape) < 2:
        B = B.reshape(-1, 1)

    if mode == "sequential":
        return _sequential_information_imbalance(A, B, metric=metric)
    elif mode == "vectorized":
        try:
            return _vectorized_information_imbalance(A, B, metric=metric)
        except MemoryError as e:
            print(e)
            print("Switching to sequential mode...")
            return _sequential_information_imbalance(A, B, metric)
    else:
        raise RuntimeError(
            f'mode {mode} not recognized. Specify "vectorized" or "sequential".'
        )


# =============================================================================
# Feature Selection related functions
# =============================================================================


# TODO: there is room here for a parallelized version of the algorithm, as for
#       each subset of features computing the information imbalance is a completely
#       independent task (embarassingly parallel). This is particularly true for
#       the sequential version, as the vectorized one may incur in memory problems
#       when multiple processes are active.
def _feature_selection(features, max_feats=5, metric="euclidean", mode="vectorized"):
    """
    Arguments
    ---------
    features     : ndarray, (num_samples, num_features)
                 Initial set of features
    max_feats    : int
                 Maximum features to be selected
    metric       : str
                 Metric used to compute the distances. Must be one of the allowed
                 metrics in the `cdist` function of SciPy (the one used to actually
                 compute distances).
                 See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html
    mode         : str
                 How the Information Imbalance is computed.
                 * 'vectorized' is faster but needs more memory. If it fails,
                   it switches to the 'sequential' mode.
                 * 'sequential' is slower, but should not run into problems when
                   the amount of data is large.
    Returns
    -------
    selected_indices  : list of int
                      Indices of the selected features
    imbalances        : dict
                      Dictionary storing the imbalances for each
                      number of retained features. Structured as:
                      imbalances   --- 0 --- 'A_B' : values
                                             'B_A' : values
                                   --- 1 --- 'A_B' : values
                                             'B_A' : values
                                   ...
    """
    # Helpers: selected columns will be deleted from these arrays
    subfeatures = features.copy()
    featindices = np.arange(features.shape[1])
    # Outputs
    imbalances = dict()
    selected = None
    selected_indices = []
    # Iterate over the number of features to be selected
    for m in range(max_feats):
        imbalances[m] = defaultdict(list)
        iterator = tqdm(range(features.shape[1] - m), desc=f"Subset d={m+1}")
        for i in iterator:
            if selected is None:
                subset = subfeatures[:, i]
            else:
                subset = np.c_[selected, subfeatures[:, i]]
            A_B = information_imbalance(subset, features, metric=metric, mode=mode)
            B_A = information_imbalance(features, subset, metric=metric, mode=mode)
            imbalances[m]["A_B"].append(A_B)
            imbalances[m]["B_A"].append(B_A)
        # Most informative feature for going from A to B
        f = np.argmin(imbalances[m]["A_B"])
        selected_indices.append(featindices[f])
        if selected is None:
            selected = subfeatures[:, f]
        else:
            selected = np.c_[selected, subfeatures[:, f]]
        # Delete the corresponding entry
        subfeatures = np.delete(subfeatures, f, axis=1)
        featindices = np.delete(featindices, f)
    return selected_indices, imbalances


def feature_selection(
    features_dict, max_feats=5, metric="euclidean", mode="vectorized"
):
    """
    Performs a greedy feature selection as described in Glielmo et al. [1]
    The feature selection is based on the information imbalance criterion.
    Arguments
    ---------
    features_dict : dict
                  Initial set of features.
                  Dictionary with key=feature_name and value=feature_values
    max_feats     : int
                  Maximum features to be selected
    metric        : str
                  Metric used to compute the distances. Must be one of the allowed
                  metrics in the `cdist` function of SciPy (the one used to actually
                  compute distances).
                  See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html
    mode          : str
                  How the Information Imbalance is computed.
                  * 'vectorized' is faster but needs more memory. If it fails,
                    it switches to the 'sequential' mode.
                  * 'sequential' is slower, but should not run into problems when
                    the amount of data is large.
    Returns
    -------
    selected_features : ndarray of str, (max_feats,)
                      Names of the selected features
    imbalances        : dict
                      Dictionary storing the imbalances for each
                      number of retained features. Structured as:
                      imbalances   --- 0 --- 'A_B' : values
                                             'B_A' : values
                                   --- 1 --- 'A_B' : values
                                             'B_A' : values
                                   ...
    """
    feat_names = np.asarray(list(features_dict.keys()))
    features = np.asarray([v for v in features_dict.values()]).T
    selected_indices, imbalances = _feature_selection(
        features, max_feats=max_feats, metric=metric, mode=mode
    )
    selected_features = feat_names[np.asarray(selected_indices)]
    return selected_features, imbalances


# =============================================================================
# Plotting related functions
# =============================================================================


def set_equalaxes(ax):
    minx, maxx = ax.get_xlim()
    miny, maxy = ax.get_ylim()
    minlim = minx if minx < miny else miny
    maxlim = maxx if maxx > maxy else maxy
    ax.set_xlim(minlim, maxlim)
    ax.set_ylim(minlim, maxlim)


def plot_imbalances(imbalances, logplot=False):
    """
    Information Imbalance Plane plot, see [1].
    Arguments
    ---------
    imbalances   : dict
                 Dictionary storing the imbalances for each
                 number of retained features. Structured as:
                 imbalances   --- 0 --- 'A_B' : values
                                        'B_A' : values
                              --- 1 --- 'A_B' : values
                                        'B_A' : values
                              ...
    Returns
    -------
    fig    : matplotlib.figure.Figure
    ax     : matplotlib.axes._subplots.AxesSubplot
    """
    fig, ax = plt.subplots(1, 1, figsize=(4, 4), dpi=150)
    ax.grid(which="both", lw=0.2)
    if logplot:
        ax.loglog()
    for k in imbalances.keys():
        plt.scatter(
            imbalances[k]["B_A"],
            imbalances[k]["A_B"],
            alpha=0.2,
            c="C%d" % k,
            label="d=%d" % (k + 1),
        )
    ax.plot([0, 1], [0, 1], ls="dashed", c="k", transform=ax.transAxes)
    set_equalaxes(ax)
    ax.legend()
    [b.set_linewidth(2) for b in ax.spines.values()]
    ax.set_xlabel(r"$\Delta(B \rightarrow A)$")
    ax.set_ylabel(r"$\Delta(A \rightarrow B)$")
    return fig, ax
