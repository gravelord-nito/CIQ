import numpy as np

def L2_norm(X, centroids):
    """
    Compute the L2 norm (Euclidean distance) between each point in X and the centroids.

    Parameters:
    X (numpy.ndarray): Input data points.
    centroids (numpy.ndarray): Current centroids.

    Returns:
    numpy.ndarray: Matrix of distances between each point and each centroid.
    """
    distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
    return distances

def cluster(X, centroids):
    """
    Assign each point to the nearest centroid.

    Parameters:
    X (numpy.ndarray): Input data points.
    centroids (numpy.ndarray): Current centroids.

    Returns:
    numpy.ndarray: Array of cluster indices for each point.
    """
    distances = L2_norm(X, centroids)
    return np.argmin(distances, axis=1)

class KMeansV5:
    """
    A hybrid version of the KMeans algorithm integrating the Artificial Bee Colony (ABC) optimization approach.

    This class applies a bee-inspired optimization strategy to improve clustering performance 
    by finding better centroids through exploration and exploitation phases.

    Attributes:
    K (int): Number of clusters.
    PKM (float): Probability of running KMeans during each bee phase iteration.
    P (int): Number of food sources (solutions).
    max_iter (int): Maximum number of iterations for the bee optimization process.
    trial_limit (int): Maximum number of unsuccessful trials before a scout bee is sent.
    verbose (bool): Whether to print detailed logs during the optimization process.
    log_file_name (str or None): Optional file name for logging purposes.
    """

    def __init__(self, K=16, KM_max_iter=3, PKM=0.1, P=10, max_iter=10, trial_limit=5, verbose=False, log_file_name=None):
        """
        Initialize the KMeansV5 class.

        Parameters:
        K (int): Number of clusters (default is 16).
        KM_max_iter (int): Maximum iterations for KMeans during internal calls (default is 3).
        PKM (float): Probability of applying KMeans to improve a solution (default is 0.1).
        P (int): Number of food sources (potential solutions) (default is 10).
        max_iter (int): Maximum iterations for the optimization process (default is 10).
        trial_limit (int): Number of consecutive unsuccessful trials allowed before a scout bee is activated (default is 5).
        verbose (bool): Enable verbose output for debugging and detailed logs (default is False).
        log_file_name (str or None): Optional file name for logging (default is None).
        """
        self.K = K
        self.PKM = PKM
        self.P = P
        self.max_iter = max_iter
        self.trial_limit = trial_limit
        self.verbose = verbose
        self.log_file_name = log_file_name

        self.food_sources = None  # P sets of centroids
        self.fitnesses = None  # Fitness value for each solution
        self.trials = None  # Unsuccessful trials for each solution

        self.best_fitness = 0
        self.best_solution = None

    def fitness(self, X, centroids):
        """
        Calculate the fitness of a solution based on the inverse of distortion.

        Parameters:
        X (numpy.ndarray): Input data points.
        centroids (numpy.ndarray): Current centroids.

        Returns:
        float: Fitness value (higher is better).
        """
        labels = cluster(X, centroids)
        distortion = 0.0
        for i in range(len(centroids)):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                distortion += np.sum(np.linalg.norm(cluster_points - centroids[i], axis=1) ** 2)
        return 1 / (1 + distortion)

    def initialize(self, X):
        """
        Initialize food sources (potential solutions) and their fitness values.

        Parameters:
        X (numpy.ndarray): Input data points.
        """
        if self.verbose:
            print(f"Initializing {self.P} food sources (solutions) randomly")

        self.food_sources = [X[np.random.choice(X.shape[0], self.K, replace=False)] for _ in range(self.P)]
        self.fitnesses = [self.fitness(X, fs) for fs in self.food_sources]
        self.trials = np.zeros(self.P)

    def mutate_solution(self, i):
        """
        Mutate a given solution by introducing variation to explore the search space.

        Parameters:
        i (int): Index of the solution to mutate.

        Returns:
        numpy.ndarray: Mutated solution.
        """
        solution = self.food_sources[i]
        neighbor = self.food_sources[np.random.choice([j for j in range(self.P) if j != i])]
        phi = np.random.uniform(-1, 1, solution.shape)
        mutated_solution = solution + phi * (solution - neighbor)
        return mutated_solution

    def apply_kmeans(self, X, i):
        """
        Apply KMeans to refine a solution with a certain probability.

        Parameters:
        X (numpy.ndarray): Input data points.
        i (int): Index of the solution to refine.
        """
        if np.random.rand() < self.PKM:
            if self.verbose:
                print("Bees considered applying k-means to improve current solution")

            kmeans = KMeansV1(K=self.K, init=self.food_sources[i], tol=0, max_iter=5)
            kmeans.fit(X)
            self.food_sources[i] = kmeans.centroids

    def employed_bee_phase(self, X):
        """
        Perform the employed bee phase, where bees search for better food sources (solutions).

        Parameters:
        X (numpy.ndarray): Input data points.
        """
        if self.verbose:
            print("Employed Bee Phase")
        for i in range(self.P):
            if self.verbose:
                print(f"Employed Bee wandering around solution {i}")

            self.apply_kmeans(X, i)
            self.fitnesses[i] = self.fitness(X, self.food_sources[i])

            new_solution = self.mutate_solution(i)
            new_fitness = self.fitness(X, new_solution)

            if new_fitness > self.fitnesses[i]:
                if self.verbose:
                    print(f"Better food source found around solution {i}")

                self.food_sources[i] = new_solution
                self.fitnesses[i] = new_fitness
                self.trials[i] = 0
            else:
                self.trials[i] += 1

    def onlooker_bee_phase(self, X):
        """
        Perform the onlooker bee phase, where onlookers select solutions to improve based on fitness.

        Parameters:
        X (numpy.ndarray): Input data points.
        """
        if self.verbose:
            print("Onlooker Bee Phase")
        probabilities = self.fitnesses / np.sum(self.fitnesses)
        for i in range(self.P):
            if np.random.rand() < probabilities[i]:
                if self.verbose:
                    print(f"Onlooker Bee chose solution {i} as a great fitness solution")

                self.apply_kmeans(X, i)
                self.fitnesses[i] = self.fitness(X, self.food_sources[i])

                new_solution = self.mutate_solution(i)
                new_fitness = self.fitness(X, new_solution)

                if new_fitness > self.fitnesses[i]:
                    if self.verbose:
                        print(f"Better food source found around solution {i}")

                    self.food_sources[i] = new_solution
                    self.fitnesses[i] = new_fitness
                    self.trials[i] = 0
                else:
                    self.trials[i] += 1

            if self.fitnesses[i] > self.best_fitness:
                if self.verbose:
                    print("Another best food source has been found!")

                self.best_fitness = self.fitnesses[i]
                self.best_solution = self.food_sources[i]

    def scout_bee_phase(self, X):
        """
        Perform the scout bee phase, where scouts find new random solutions after trial limits.

        Parameters:
        X (numpy.ndarray): Input data points.
        """
        if self.verbose:
            print("Scout Bee Phase")
        for i in range(self.P):
            if self.trials[i] >= self.trial_limit:
                if self.verbose:
                    print(f'No better food sources has been found around solution {i} for too long')
                    print(f'A scout bee is being sent to find another food source randomly')
                self.food_sources[i] = X[np.random.choice(X.shape[0], self.K, replace=False)]
                self.trials[i] = 0

    def fit(self, X):
        """
        Fit the model to the data using the bee-inspired optimization process.

        Parameters:
        X (numpy.ndarray): Input data points.

        Returns:
        numpy.ndarray: Optimized centroids.
        """
        self.initialize(X)
        for epoch in range(self.max_iter):
            if self.verbose:
                print(f"Bee phase iteration {epoch}")
            self.employed_bee_phase(X)
            self.onlooker_bee_phase(X)
            self.scout_bee_phase(X)
        self.centroids = self.best_solution
        return self.centroids

    def predict(self, X):
        """
        Predict the closest cluster for each point in X based on the optimized centroids.

        Parameters:
        X (numpy.ndarray): Input data points.

        Returns:
        numpy.ndarray: The index of the nearest centroid for each point.
        """
        dist = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(dist, axis=1)

    def distortion(self, X):
        """
        Calculate the total distortion (sum of squared distances to nearest centroids).

        Parameters:
        X (numpy.ndarray): Input data points.

        Returns:
        float: Total distortion value.
        """
        centroids = self.centroids
        labels = cluster(X, centroids)
        distortion = 0.0
        for i in range(len(centroids)):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                distortion += np.sum(np.linalg.norm(cluster_points - centroids[i], axis=1) ** 2)
        return distortion
