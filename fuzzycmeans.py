import numpy as np
from kmeans.py import KMeansV3

class KMeansV4(KMeansV3):
    """
    An extended version of the KMeansV3 class that implements Fuzzy C-Means clustering.

    This class introduces the concept of "fuzziness," where data points can belong to multiple clusters with different degrees of membership, rather than being assigned to a single cluster.

    Inherits:
    KMeansV3: The KMeans implementation with centroid rerolling and k-means++ initialization.
    """

    def __init__(self, fuzziness=2, n_reroll=0, n_worst_reroll=1, *args, **kwargs):
        """
        Initialize the KMeansV4 class.

        Parameters:
        fuzziness (float): The fuzziness parameter that controls the degree of cluster overlap.
                           Must be greater than 1; higher values lead to more overlap.
        n_reroll (int): The number of times to reroll (reselect) the centroids (inherited, default is 0).
        n_worst_reroll (int): The number of worst points to consider for rerolling (inherited, default is 1).
        *args: Additional positional arguments for the superclass.
        **kwargs: Additional keyword arguments for the superclass.
        """
        super().__init__(*args, **kwargs)
        self.fuzziness = fuzziness  # fuzziness parameter

    def get_membership(self, X):
        """
        Calculate the membership matrix for each data point to each cluster.

        The membership value indicates the degree to which each point belongs to each cluster, 
        determined by the inverse distance to the centroids raised to the power defined by the fuzziness parameter.

        Parameters:
        X (numpy.ndarray): Input data points.

        Returns:
        numpy.ndarray: Membership matrix where each row represents a point and each column represents a cluster.
        """
        distances = self.transform(X)
        distances = np.fmax(distances, np.finfo(np.float64).eps)  # Avoid division by zero
        inv_dist = 1 / distances
        inv_dist = inv_dist ** (2 / (self.fuzziness - 1))
        membership = inv_dist / np.sum(inv_dist, axis=1, keepdims=True)
        return membership

    def update_membership(self, X):
        """
        Update the membership matrix and check for convergence.

        The method computes the new membership matrix and checks whether the change 
        is within the specified tolerance to determine if the algorithm has converged.

        Parameters:
        X (numpy.ndarray): Input data points.

        Returns:
        bool: True if the change in membership is within the tolerance; False otherwise.
        """
        new_membership = self.get_membership(X)
        if np.linalg.norm(new_membership - self.membership) <= self.tol:
            self.membership = new_membership
            return True
        else:
            self.membership = new_membership
            return False

    def init_centroids(self, X):
        """
        Initialize centroids and calculate the initial membership matrix.

        Parameters:
        X (numpy.ndarray): Input data points.
        """
        super().init_centroids(X)
        self.membership = self.get_membership(X)

    def update_centroids(self, X):
        """
        Update the centroids based on the membership matrix.

        The centroids are updated by weighting the points according to their membership values raised 
        to the power defined by the fuzziness parameter.

        Parameters:
        X (numpy.ndarray): Input data points.

        Returns:
        bool: True if the membership matrix converges; False otherwise.
        """
        um = self.membership ** self.fuzziness
        self.centroids = (um.T @ X) / np.sum(um.T, axis=1, keepdims=True)
        return self.update_membership(X)

    def predict(self, X):
        """
        Predict the closest cluster for each point in X based on membership values.

        Parameters:
        X (numpy.ndarray): Input data points.

        Returns:
        numpy.ndarray: The index of the nearest centroid for each point.
        """
        return self.get_membership(X).argmax(axis=1)

