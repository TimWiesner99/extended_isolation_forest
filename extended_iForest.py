"""
Implementation of the extended isolation forest based on:

    S. Hariri, M. Carrasco Kind and R. J. Brunner, "Extended Isolation Forest,"
    in IEEE Transactions on Knowledge and Data Engineering, doi: 10.1109/TKDE.2019.2947676.

Isolation forests are based on the assumption that anomalies (or "outliers") are 'few and different', which makes them
more susceptible to isolation than normal instances. In each node of a tree in the isolation forest, the data gets split
into two parts. Because of their susceptibility to isolation, anomalous points are closer to the root of the tree.
In contrast to other anomaly-detection-approaches, isolation trees are model-free. The 'extended' variant of isolation
trees, that is implemented here, splits the data with hyperplanes, instead of splitting on one parameter/dimension at a
time (like regular isolation trees).
"""

__author__ = "Tim Wiesner, Anwar Cruz"
__version__ = "1.0"
__email__ = "tim200799@gmail.com"

import math
import random
import numpy as np


# CLASSES ################
class iForest:
    """
    Class containing the extended isolation forest.

    Attributes
    ----------
    data : np.array
        a numpy array containing the data that the trees in the forest are fitted on
    trees : Array of iTree objects
        the array containing the individual iTree objects
    subsample_size : int
        The size of the subsamples that are passed to the individual trees. Default size 256 was
        empirically shown to be relatively optimal in the original paper.
    dims_per_cut : int
        Number indicating how many dimensions in the data should be used per cut. Using only one
        dimension at a time is equivalent to the regular, "non-extended" isolation forest. Setting this to 0
        is equivalent to the number of dimensions in the data and uses all dimensions.

    Methods
    -------
    anomaly_score(x) -> float
        Returns the anomaly score of one data point given the forest of fitted trees.
        The anomaly score is between 0 and 1, representing the probability that this point is an anomaly.

    """

    def __init__(self, data: np.array, number_of_trees: int = 100, subsample_size: int = 256, dims_per_cut: int = 0):
        """
        Initializes and fits the extended isolation forest.
        :param data: np.array
            The data that the forest will be fitted on.
        :param number_of_trees: int
            The number of trees in the forest. The default is 100.
        :param subsample_size: int
            The size of the subsamples that are passed to the individual trees. Default size 256 was
            empirically shown to be relatively optimal in the original paper.
        :param dims_per_cut: int
            Number indicating how many dimensions in the data should be used per cut. Using only one
            dimension at a time is equivalent to the regular, "non-extended" isolation forest. Setting this to 0
            is equivalent to the number of dimensions in the data and uses all dimensions.
        """

        assert data.shape[1] >= dims_per_cut >= 0, 'dimensions per cut cannot be higher than number of parameters and at least 0'
        self.data = data
        self.subsample_size = subsample_size

        if dims_per_cut == 0:
            self.dims_per_cut = self.data.shape[1]
        else:
            self.dims_per_cut = dims_per_cut

        depth_limit = math.ceil(math.log2(subsample_size))
        self.trees = []
        for i in range(number_of_trees):
            subsample = data[np.random.choice(data.shape[0], subsample_size, replace=True), :]
            self.trees.append(iTree(subsample, depth=0, depth_limit=depth_limit, dims_per_cut=dims_per_cut).fit())

    def anomaly_score(self, x: np.array):
        """
        Returns the anomaly score of one data point given the forest of fitted trees.
        The anomaly score is between 0 and 1, representing the probability that this point is an anomaly.
        :param x: np.array
            One point as an np.array
        :return: float
            Anomaly score between 0 and 1.
        """
        mean_path_length = sum([tree.path_length(x) for tree in self.trees]) / len(self.trees)
        return 2 ** (-mean_path_length / avg_path_length(self.subsample_size))


class iTree:
    """
    Class containing the extended isolation forest.

    Attributes
    ----------
    data : np.array
        a numpy array containing the data that the trees in the forest are fitted on
    dims_per_cut : int
        Number indicating how many dimensions in the data should be used per cut. Using only one
        dimension at a time is equivalent to the regular, "non-extended" isolation forest. Setting this to 0
        is equivalent to the number of dimensions in the data and uses all dimensions.
    depth : int
        Current depth of this node in the complete tree.
    depth_limit : int
        Limit to how far a tree may grow. Typically computed from the subsampling size set in the iForest class.
        After reaching the depth limit, a point is as least as "normal" as half of the other points and thus unlikely
        to be an anomaly and not interesting. That is why the construction of the tree is cut off there.

    Methods
    -------
    fit() -> iTree
        This method fits the tree and either returns a leaf node, or an interior node that contains another two iTrees
        which are then also fitted recursively.
    path_length(x) -> float
        Returns the length of the path from the root node to a leaf node at the bottom of the tree that would contain
        point x. Anomalies have a shorter path length than normal instances.
    """

    def __init__(self, data: np.array, depth: int, depth_limit: int, dims_per_cut: int = 0):
        """
        Initializes an extended isolation tree.
        :param data: np.array
            The data that the forest will be fitted on.
        :param depth : int
            Current depth of this node in the complete tree.
        :param depth_limit : int
            Limit to how far a tree may grow. Typically computed from the subsampling size set in the iForest class.
            After reaching the depth limit, a point is as least as "normal" as half of the other points and thus unlikely
            to be an anomaly and not interesting. That is why the construction of the tree is cut off there.
        :param dims_per_cut : int
            Number indicating how many dimensions in the data should be used per cut. Using only one
            dimension at a time is equivalent to the regular, "non-extended" isolation forest. Setting this to 0
            is equivalent to the number of dimensions in the data and uses all dimensions.
        """
        assert data.shape[1] >= dims_per_cut >= 0, 'dimensions per cut cannot be higher than number of parameters and at least 0'
        self.data = data
        self.depth = depth
        self.depth_limit = depth_limit

        if dims_per_cut == 0:
            self.dims_per_cut = self.data.shape[1]
        else:
            self.dims_per_cut = dims_per_cut

    def fit(self):
        """
        Fits this tree to the data. Recursively calls itself on its subtrees until the tree is fully grown.
        :return: iTree
            Another instance of an iTree that is either a leaf node or an internal node, which contains two
            instances of isolation trees, which are then fitted recursively and also turn into either leaf or
            internal nodes.
        """
        if self.depth >= self.depth_limit or len(self.data) <= 1:
            return Leaf(len(self.data))
        else:
            # create n array (slope) and p array (intercept)
            n = np.empty(self.data.shape[1])
            p = np.empty(self.data.shape[1])
            exclusion = random.sample(range(0, self.data.shape[1]), (self.data.shape[1] - self.dims_per_cut))
            for i in range(self.data.shape[1]):
                if i in exclusion:
                    n[i] = 0
                else:
                    n[i] = np.random.normal(0, 1)
                p[i] = np.random.uniform(min(self.data[:, i]), max(self.data[:, i]))
            data_l = self.data[np.dot((self.data - p), n) <= 0]
            data_r = self.data[np.dot((self.data - p), n) > 0]
            return Interior(iTree(data_l, self.depth + 1, self.depth_limit, self.dims_per_cut).fit(),
                            iTree(data_r, self.depth + 1, self.depth_limit, self.dims_per_cut).fit(), n, p)

    def path_length(self, x: np.array) -> float:
        """
        Returns the length of the path from the root node to a leaf node at the bottom of the tree that would contain
        point x. Anomalies have a shorter path length than normal instances.
        :param x: np.array
            One data point of which the path length is to be determined.
        :return: float
            The path length of that point. Might be an estimate of the point has a longer-than-average path length.
        """
        if isinstance(self, Leaf):
            return 1 + avg_path_length(self.size)
        elif isinstance(self, Interior):
            n = self.normal
            p = self.intercept
            if np.dot((x - p), n) <= 0:
                return 1 + self.left.path_length(x)
            else:
                return 1 + self.right.path_length(x)


class Interior(iTree):
    """
    An instance of iTree that contains two other instances of iTrees as children, as well as information about the cut
    through the data at this node.

    Attributes
    ----------
    left : iTree
        The "left-child" iTree containing one half of the data.
    right : iTree
        The "right-child" iTree containing the other half of the data.
    normal : np.array
        The normal vector of the hyperplane that cuts through the data in this node.
    intercept : np.array
        The intercept vector of the hyperplane that cuts through the data in this node.
    """

    def __init__(self, left: iTree, right: iTree, normal: np.array, intercept: np.array):
        """
        Initializes an interior node.
        :param left : iTree
            The "left-child" iTree containing one half of the data.
        :param right : iTree
            The "right-child" iTree containing the other half of the data.
        :param normal : np.array
            The normal vector of the hyperplane that cuts through the data in this node.
        :param intercept : np.array
            The intercept vector of the hyperplane that cuts through the data in this node.
        """
        self.left = left
        self.right = right
        self.normal = normal
        self.intercept = intercept


class Leaf(iTree):
    """
    An instance of iTree at the bottom of a tree. Because branches of isolation trees that have already reached the
    depth limit (equivalent to average depth of a tree containing that many data points) can contain multiple data points,
    the number of left-over data points is saved as an attribute.

    Attributes
    ----------
    size : int
        The number of data points that are left in this leaf node.
    """

    def __init__(self, size: int):
        """
        Initializes a leaf node.
        :param size : int
            The number of data points that are left in this leaf node.
        """
        self.size = size


# FUNCTIONS ################
def avg_path_length(size: int, estimate_harmonic=True) -> float:
    """
    Computes the average path length of a point in an isolation tree containing n data points.
    :param size: int
        Size n of data points in the tree.
    :param estimate_harmonic: bool
        Boolean indicating whether the harmonic number should be estimated or not. Default is True because computing
        the harmonic number can only be done recursively which is unreasonable for anything but very small trees.
    :return: float
        The average path length of a point in an isolation tree containing n data points.
    """
    def harmonic(n) -> float:
        if estimate_harmonic:
            return np.log(size) + np.euler_gamma
        else:
            if n > 1:
                return 1 / n + harmonic(n - 1)
            else:
                return 1

    if size == 0:
        return 0
    else:
        return 2 * harmonic(size - 1) - (2 * (size - 1) / (size + 1))
