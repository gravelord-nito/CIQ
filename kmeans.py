import numpy as np

def euc_distance(x, y):
    """
    Calculate the Euclidean distance between two points.

    Parameters:
    x (numpy.ndarray): The first point.
    y (numpy.ndarray): The second point.

    Returns:
    numpy.ndarray: The Euclidean distance between the two points.
    """
    return np.sqrt(np.sum((x - y) ** 2, axis=2))

class KMeansV1:
    """
    A basic implementation of the KMeans clustering algorithm.

    Attributes:
    K (int): The number of clusters.
    tol (float): The tolerance to declare convergence.
    max_iter (int): The maximum number of iterations.
    verbose (bool): If True, prints detailed logs during execution.
    log_file_name (str): The name of the file where log is saved.
    centroids (numpy.ndarray): The coordinates of the cluster centroids.
    """

    def __init__(self, K=16, init=None, tol=1e-5, max_iter=25, verbose=False, log_file_name=None):
        """
        Initialize the KMeansV1 class.

        Parameters:
        K (int): The number of clusters.
        init (numpy.ndarray or None): Initial centroids or None to randomly initialize.
        tol (float): The tolerance to declare convergence.
        max_iter (int): The maximum number of iterations.
        verbose (bool): If True, prints detailed logs during execution.
        log_file_name (str): The name of the file where log is saved.
        """
        self.K = K
        self.tol = tol
        self.max_iter = max_iter
        self.verbose = verbose
        self.log_file_name = log_file_name
        self.centroids = None if init is None else init.copy()

    def init_centroids(self, X):
        """
        Initialize the centroids randomly from the input data.

        Parameters:
        X (numpy.ndarray): Input data points.
        """
        if self.verbose:
            print("Initializing centroids")

        random_idx = np.random.choice(X.shape[0], self.K, replace=False)
        self.centroids = X[random_idx]

    def transform(self, X):
        """
        Calculate the Euclidean distances from data points to centroids.

        Parameters:
        X (numpy.ndarray): Input data points.

        Returns:
        numpy.ndarray: The distances of each point to the centroids.
        """
        return np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)

    def cluster(self, X):
        """
        Assign each data point to the nearest centroid.

        Parameters:
        X (numpy.ndarray): Input data points.

        Returns:
        list: A list of clusters with the points assigned to each cluster.
        """
        clusters = [[] for _ in range(self.K)]
        distances = self.transform(X)
        for i, point in enumerate(X):
            clusters[np.argmin(distances[i])].append(point)
        return clusters

    def update_centroids(self, X):
        """
        Update the centroids based on the mean of the points assigned to each cluster.

        Parameters:
        X (numpy.ndarray): Input data points.

        Returns:
        bool: True if centroids converge (change is within tolerance), False otherwise.
        """
        clusters = self.cluster(X)
        new_centroids = []
        for cluster in clusters:
            if len(cluster):
                new_centroids.append(np.mean(cluster, axis=0))
            else:
                random_idx = np.random.randint(0, X.shape[0])
                new_centroids.append(X[random_idx])
        new_centroids = np.array(new_centroids)

        if ((self.centroids - new_centroids)**2).sum() <= self.tol:
            self.centroids = new_centroids
            return True
        else:
            self.centroids = new_centroids
            return False

    def L2_norms(self, X):
        """
        Calculate the L2 norms (distances) of each point from its nearest centroid.

        Parameters:
        X (numpy.ndarray): Input data points.

        Returns:
        numpy.ndarray: The L2 norms of the points from their nearest centroids.
        """
        distances = self.transform(X)
        nearest_centroid_idx = np.argmin(distances, axis=1)
        return np.linalg.norm(X - self.centroids[nearest_centroid_idx], axis=1)

    def distortion(self, X):
        """
        Calculate the sum of squared L2 norms, used as an evaluation metric.

        Parameters:
        X (numpy.ndarray): Input data points.

        Returns:
        float: The sum of squared L2 norms.
        """
        return np.sum(self.L2_norms(X) ** 2)

    def log(self, X):
        """
        Log the L2 norms to a file.

        Parameters:
        X (numpy.ndarray): Input data points.
        """
        file_path = os.path.join('logs', self.log_file_name + '.txt')
        arr_str = np.array2string(self.L2_norms(X), separator=',', precision=2)[1:-1]
        with open(file_path, 'a') as file:
            file.write(arr_str + '\n')

    def converge(self, X):
        """
        Run the iterative process to converge the centroids.

        Parameters:
        X (numpy.ndarray): Input data points.
        """
        if self.verbose:
            print("Converging the centroids")

        for epoch in range(self.max_iter):
            if self.verbose:
                print(f"Epoch {epoch}")

            if self.update_centroids(X):
                break
            if self.log_file_name is not None:
                self.log(X)

    def fit(self, X):
        """
        Fit the KMeans model to the input data.

        Parameters:
        X (numpy.ndarray): Input data points.
        """
        if self.centroids is None:
            self.init_centroids(X)
        self.converge(X)

    def predict(self, X):
        """
        Predict the closest cluster for each point in X.

        Parameters:
        X (numpy.ndarray): Input data points.

        Returns:
        numpy.ndarray: The index of the nearest centroid for each point.
        """
        distances = self.transform(X)
        return np.argmin(distances, axis=1)


class KMeansV2(KMeansV1):
    """
    An extended version of the KMeansV1 class with a modified centroid initialization method.

    This class implements the k-means++ initialization to improve the convergence speed
    of the KMeans algorithm.

    Inherits:
    KMeansV1: The base KMeans implementation class.
    """

    def init_centroids(self, X):
        """
        Initialize the centroids using the k-means++ algorithm.

        The first centroid is chosen randomly from the input data points.
        Each subsequent centroid is chosen with a probability proportional
        to the squared distance from the nearest already chosen centroid.

        Parameters:
        X (numpy.ndarray): Input data points.
        """
        if self.verbose:
            print("Initializing centroids with k-means++")

        n = X.shape[0]
        self.centroids = []

        # Choose the first centroid randomly from the data points.
        self.centroids.append(X[np.random.choice(n)])

        # Choose the remaining K-1 centroids.
        for i in range(self.K - 1):
            # Compute distances from each point to the nearest centroid.
            distances = self.transform(X)
            nearest_d2 = np.min(distances, axis=1) ** 2

            # Compute the probabilities proportional to the squared distances.
            total_prob = np.sum(nearest_d2)
            probs = nearest_d2 / total_prob

            # Choose the next centroid based on computed probabilities.
            self.centroids.append(X[np.random.choice(n, p=probs)])

        self.centroids = np.array(self.centroids)


class KMeansV3(KMeansV2):
    """
    An extended version of the KMeansV2 class with an additional centroid refinement mechanism.

    This class refines the centroids found by the k-means++ algorithm by rerolling
    (reselecting) the worst centroids to minimize distortion further.

    Inherits:
    KMeansV2: The extended KMeans implementation with k-means++ initialization.
    """

    def __init__(self, n_reroll=3, n_worst_reroll=5, *args, **kwargs):
        """
        Initialize the KMeansV3 class.

        Parameters:
        n_reroll (int): The number of times to reroll (reselect) the centroids.
        n_worst_reroll (int): The number of worst points to consider for rerolling.
        *args: Additional positional arguments for the superclass.
        **kwargs: Additional keyword arguments for the superclass.
        """
        super().__init__(*args, **kwargs)
        self.n_reroll = n_reroll
        self.n_worst_reroll = n_worst_reroll

    def reroll(self, X):
        """
        Refine the centroids by rerolling the worst centroids.

        The function identifies the worst-performing centroid and attempts to replace it 
        with a better candidate from the data points to minimize the distortion.

        Parameters:
        X (numpy.ndarray): Input data points.
        """

        def worst_cent():
            """
            Identify the worst-performing centroid.

            Returns:
            int: The index of the worst-performing centroid.
            """
            cent_distances = self.transform(self.centroids)
            # Set diagonal to infinity to exclude self-distances.
            cent_distances = cent_distances + np.diag(np.repeat(np.inf, cent_distances.shape[0]))
            mnn = np.argmin(cent_distances)
            i, j = np.unravel_index(mnn, cent_distances.shape)
            return i if np.mean(cent_distances[i]) < np.mean(cent_distances[j]) else j

        def best_swap():
            """
            Find the best candidate point to replace the worst-performing centroid.

            Returns:
            int: The index of the data point to swap in as a new centroid.
            """
            distances = self.transform(X)
            mxx = np.argpartition(distances.flatten(), -self.n_worst_reroll)[-self.n_worst_reroll:]
            mxx = np.random.choice(mxx)
            i, j = np.unravel_index(mxx, distances.shape)
            return i

        if self.verbose:
            print('Rerolling centroids')
        
        best_centroids = self.centroids.copy()
        best_distortion = self.distortion(X)
        
        for reroll_epoch in range(self.n_reroll):
            if self.verbose:
                print(f'Reroll epoch {reroll_epoch}')

            # Replace the worst centroid with the best candidate.
            self.centroids[worst_cent()] = X[best_swap()]
            self.converge(X)

            distortion = self.distortion(X)
            if distortion < best_distortion:
                if self.verbose:
                    print('Better centroids found!')
                best_centroids = self.centroids.copy()
                best_distortion = distortion

        self.centroids = best_centroids

    def fit(self, X):
        """
        Fit the KMeans model to the input data and refine the centroids.

        Parameters:
        X (numpy.ndarray): Input data points.
        """
        super().fit(X)
        self.reroll(X)

