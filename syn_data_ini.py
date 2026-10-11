import numpy as np
from typing import Tuple

def k_medoids_sampling(distance_matrix: np.ndarray, 
                      k: int = 100, 
                      max_iters: int = 100,
                      random_state: int = None, 
                      if_return_label: bool = False) :

    if random_state is not None:
        np.random.seed(random_state)
    
    n_points = distance_matrix.shape[0]
    
    medoids = np.random.choice(n_points, k, replace=False)
    # medoids_history = [medoids.copy()]
    
    print(f"Starting k-centroid sampling: selecting {k} points from {n_points} points.")
    
    for iteration in range(max_iters):
        distances_to_medoids = distance_matrix[:, medoids]
        labels = np.argmin(distances_to_medoids, axis=1)

        new_medoids = np.zeros(k, dtype=int)
        
        for i in range(k):
            cluster_points = np.where(labels == i)[0]
            
            if len(cluster_points) == 0:
                new_medoids[i] = np.random.choice(n_points)
                continue

            cluster_distances = distance_matrix[cluster_points][:, cluster_points]
            total_distances = cluster_distances.sum(1)
            
            min_index = np.argmin(total_distances)
            new_medoids[i] = cluster_points[min_index]
        
        if np.array_equal(np.sort(medoids), np.sort(new_medoids)):
            print(f"The algorithm converges after {iteration+1} iterations.")
            break
        
        medoids = new_medoids
        # medoids_history.append(medoids.copy())
        
        if (iteration + 1) % 10 == 0:
            current_cost = calculate_total_cost(distance_matrix, medoids, labels)
            print(f"Iteration {iteration+1}, current cost: {current_cost:.4f}")
    
    if not if_return_label:
        return medoids.astype(int)
    else:
        distances_to_medoids = distance_matrix[:, medoids]
        final_labels = np.argmin(distances_to_medoids, axis=1)
        return medoids, final_labels.numpy().tolist()
    # return medoids, final_labels, medoids_history

def calculate_total_cost(distance_matrix: np.ndarray, 
                        medoids: np.ndarray, 
                        labels: np.ndarray) -> float:
    total_cost = 0.0
    for i, medoid in enumerate(medoids):
        cluster_points = np.where(labels == i)[0]
        if len(cluster_points) > 0:
            total_cost += np.sum(distance_matrix[cluster_points, medoid])
    return total_cost