import numpy as np
import matplotlib.pyplot as plt

class Kmeans:
    def __init__(self, K_clusters = 3, n_iter = 500):
        self.K = K_clusters
        self.n = n_iter
        self.centroids = None
        self.labels = None

    def euclidean_distance(self, point1, point2): #finds the straight line distance between points
        return np.sqrt(np.sum((point1 - point2)**2))
    
    def fit(self, X):
        n_samp, n_feat = X.shape

        random_indices = np.random.choice(n_samp, self.K, replace = False) #random initialization of centroids
        self.centroids = X[random_indices]

        print(f"Initialized {self.K} random centroids")

        for iteration in range(self.n):
            self.labels = self._assign_clusters(X) 
            old_centroids = self.centroids.copy()
            self.centroids = self._update_centroids(X)

            if self._has_converged(old_centroids):
                print(f"Converged after {iteration +1} iterations")
                break
            if iteration % 10 == 0:
                print(f"Iteration {iteration}")
        return self
    
    def _assign_clusters(self, X):
        n_samples = X.shape[0]
        distances = np.zeros((n_samples, self.K))

        for i, centroid in enumerate(self.centroids):
            distances[:, i] = np.sqrt(np.sum((X-centroid)**2, axis =1))
        return np.argmin(distances, axis = 1)
    def _update_centroids(self,X):
        new_centroids = np.zeros((self.K, X.shape[1]))

        for cluster_index in range(self.K):
            cluster_points = X[self.labels == cluster_index]
            if len(cluster_points) > 0:
                new_centroids[cluster_index] = np.mean(cluster_points, axis=0)

            else:
                new_centroids[cluster_index] = self.centroids[cluster_index]
        return new_centroids
    def _has_converged(self, old_centroids):
        #checks if distances residuals between centroid is very close to 0
        distances = [self.euclidean_distance(self.centroids[i], old_centroids[i]) for i in range(self.K)]
        return np.all(np.array(distances) < 1e-10)
    
    def predict(self,X):
        return self._assign_clusters(X)
    def calculate_inertia(self, X):
        inertia = 0

        for cluster_index in range(self.K):
            cluster_points = X[self.labels == cluster_index]
            if len(cluster_points)> 0:
                centroid = self.centroids[cluster_index]

                for point in cluster_points:
                    distance = self.euclidean_distance(point, centroid)
                    inertia += distance **2
            return inertia
        
            

if __name__ == "__main__":
    #Visualize elbow point to find optimal amount of clusters
    k_range = range(1, 11)
    inertias = []
    np.random.seed(42)
    print("\n making test data...")
    cluster_1 = np.random.randn(50,2) *0.5 + [0,5]
    cluster_2 = np.random.randn(50,2) * 0.5 + [5,5]
    cluster_3 = np.random.randn(50,2) * 0.5 + [5,0]

    X = np.vstack([cluster_1, cluster_2, cluster_3])

    print("\nStarting Elbow Method analysis...")
    for k in k_range:
        model = Kmeans(K_clusters=k, n_iter=300)
        model.fit(X)
        inertias.append(model.calculate_inertia(X))
        print(f"K={k} | Inertia: {inertias[-1]:.2f}")

    plt.figure(figsize=(8, 5))
    plt.plot(k_range, inertias, marker='o', linestyle='--', color='b')
    plt.title('Elbow Method for Optimal K', fontsize=14)
    plt.xlabel('Number of Clusters (K)', fontsize=12)
    plt.ylabel('Inertia', fontsize=12)
    plt.xticks(k_range)
    plt.grid(True, alpha=0.3)


    plt.show()
    print("\nK Means testing\n")


    print(f"Created {len(X)} points in 3 clusters")
    print(f"Shape = {X.shape}")
    kmeans = Kmeans()
    kmeans.fit(X)

    print("\nResults\n")
    print("Final centroids:")
    for i, centroid in enumerate(kmeans.centroids):
        print(f"Cluster {i}: [{centroid[0], [centroid[1]]}]")

    print(f"Inertia (lower is better): {kmeans.calculate_inertia(X)}")
    
    print("\nPoints per clusters:")
    for i in range(3):
        count = np.sum(kmeans.labels == i)
        print(f"Cluster {i}: {count} points")
    #plot kmeans clustering against sample data
    plt.figure(figsize=(10, 8))
    colors = ['red', 'blue', 'green']
    
    for cluster_index in range (3):
        cluster_points = X[kmeans.labels == cluster_index]
        plt.scatter(cluster_points[:, 0], cluster_points[:,1],
                    c=colors[cluster_index],
                    label=f'Cluster {cluster_index}',
                    alpha=0.6,
                    edgecolors='k',
                    s=50
                    ) 
    plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:,1],
                c='black',
                marker="X",
                s=300,
                linewidths=3,
                label='Centroids',
                edgecolors='yellow',
                zorder=10)
    plt.xlabel('Feature 1', fontsize=12)
    plt.ylabel('Feature 2', fontsize=12)
    plt.title('K-means clustering results', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.show()