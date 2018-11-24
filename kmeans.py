import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

def compute_euclidean_distance(point, centroid):
    return np.sqrt(np.sum((point - centroid)**2))

def assign_label_cluster(distance, data_point, centroids):
    index_of_minimum = min(distance, key=distance.get)
    return [index_of_minimum, data_point, centroids[index_of_minimum]]

def compute_new_centroids(cluster_label, centroids):
    return np.array(cluster_label + centroids)/2



def print_label_data(result):
    print("Result of k-Means Clustering: \n")
    for data in result[0]:
        print("data point: {}".format(data[1]))
        print("cluster number: {} \n".format(data[0]))
    print("Last centroids position: \n {}".format(result[1]))

def create_centroids(lo, hi, k):
    centroids = []
    for _ in xrange(k):
        centroids.append([])
        for i in xrange(len(lo)):
            num = (hi[i] - lo[i]) * np.random.random() + lo[i]
            centroids[-1].append(num)
    return np.array(centroids)

if __name__ == "__main__":
    #filename = os.path.dirname(__file__) + "./data.csv"
    
    df = pd.read_csv("./iris.csv")
    n, d = df.shape[0], df.shape[1] - 1


    #data_points = np.genfromtxt(filename, delimiter=",")
    
    
    niter = 30
    nk = 3
    
    centroids = create_centroids([min(df.iloc[:,i]) for i in xrange(d)], [max(df.iloc[:,i]) for i in xrange(d)], nk)
    
    for iteration in xrange(niter):
        nxt, nxt_count = [[0] * d for _ in xrange(nk)], [0] * nk
        
        for i in xrange(n):
            min_dis, label = float("Inf"), -1
            for j in range(nk):
                dis = np.linalg.norm(centroids[j] - df.iloc[i].tolist()[:-1])
                if dis < min_dis:
                    min_dis, label = dis, j
            
            for k in xrange(d):
                nxt[label][k] = (nxt[label][k] * nxt_count[label] + df.iloc[i][k]) / (nxt_count[label] + 1.0)
            nxt_count[label] += 1

        centroids = np.array(nxt)
        print([list(x) for x in centroids])
            #if iteration == (total_iteration - 1):
            #    cluster_label.append(label)
    
    plt.plot(df.iloc[:, 1], df.iloc[:, 2], ".")
    plt.plot(centroids[0][1], centroids[0][2], "*")
    plt.plot(centroids[1][1], centroids[1][2], "*")
    plt.plot(centroids[2][1], centroids[2][2], "*")
    plt.show()
    
    #print_label_data([cluster_label, new_centroids])
    print()
    
