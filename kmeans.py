import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def kmeans(df, nk, niter = 100, tol = 1e-6):
    
    # size of data
    n, d = df.shape[0], df.shape[1]
    
    # initialize randomly
    """
    lo = [min(df.iloc[:,i]) for i in xrange(d)]
    hi = [max(df.iloc[:,i]) for i in xrange(d)]
    centroids = []
    for _ in xrange(nk):
        centroids.append([])
        for i in xrange(len(lo)):
            num = (hi[i] - lo[i]) * np.random.random() + lo[i]
            centroids[-1].append(num)
    centroids = np.array(centroids)
    """
    
    # k-means++
    p = np.zeros(n)
    index = [np.random.choice(n)]
    centroids = [df.iloc[index[-1]].tolist()]
    for _ in xrange(nk - 1):
        for i in xrange(n):
            min_dis, label = float("Inf"), -1
            for j in range(len(index)):
                dis = np.linalg.norm(centroids[j] - df.iloc[i])
                if dis < min_dis:
                    min_dis, label = dis, j
            p[i] = min_dis * min_dis if not label in index else 0.0
        p = p / sum(p)
        index.append(np.random.choice(n, 1, p=p)[0])
        centroids.append(df.iloc[index[-1]].tolist())
    centroids = np.array(centroids)
    
    # some needed arrays
    err = []
    
    # main loop
    for iteration in xrange(niter):
        
        groups = [[] for _ in xrange(nk)]
        nxt = np.array([[0.0] * d for _ in xrange(nk)])
        nxt_count = [0.0] * nk
        error = 0.0
        
        # for each data
        for i in xrange(n):
            min_dis, label = float("Inf"), -1
            for j in range(nk):
                dis = np.linalg.norm(centroids[j] - df.iloc[i].tolist())
                if dis < min_dis:
                    min_dis, label = dis, j
            groups[label].append(i)
            error += min_dis
            
            # calculate new centroids
            nxt[label] = (nxt[label] * nxt_count[label] + df.iloc[i].tolist()) / (nxt_count[label] + 1.0)
            nxt_count[label] += 1
        
        # update centroids, error
        centroids = np.array(nxt)
        error /= n
        if err and abs(error - err[-1]) < tol:
            break
        err.append(error)
        
        # print error
        if (iteration + 1) % 1 == 0:
            print("{}: {}".format(iteration + 1, error))
    
    return centroids, groups, err

if __name__ == "__main__":
    
    # read data, initialize parameters
    df = pd.read_csv("./iris.csv")
    classes = sorted(list(set(df["class"])))
    cid = {c: i for i, c in enumerate(classes)}
    
    # ground truth
    k = 3
    real_groups = [[] for _ in xrange(k)]
    for i in xrange(len(df)):
        c = df.iloc[i]["class"]
        real_groups[cid[c]].append(i)
    
    # k = 3, show compare clustering results with actual category
    centroids, groups, err = kmeans(df.iloc[:, :-1], 3)
    
    plt.figure()
    plt.subplot(231)
    plt.plot(df.iloc[groups[0], 0], df.iloc[groups[0], 1], "r.")
    plt.plot(df.iloc[groups[1], 0], df.iloc[groups[1], 1], "b.")
    plt.plot(df.iloc[groups[2], 0], df.iloc[groups[2], 1], "g.")
    plt.subplot(232)
    plt.plot(df.iloc[groups[0], 0], df.iloc[groups[0], 2], "r.")
    plt.plot(df.iloc[groups[1], 0], df.iloc[groups[1], 2], "b.")
    plt.plot(df.iloc[groups[2], 0], df.iloc[groups[2], 2], "g.")
    plt.subplot(233)
    plt.plot(df.iloc[groups[0], 0], df.iloc[groups[0], 3], "r.")
    plt.plot(df.iloc[groups[1], 0], df.iloc[groups[1], 3], "b.")
    plt.plot(df.iloc[groups[2], 0], df.iloc[groups[2], 3], "g.")
    plt.subplot(234)
    plt.plot(df.iloc[groups[0], 1], df.iloc[groups[0], 2], "r.")
    plt.plot(df.iloc[groups[1], 1], df.iloc[groups[1], 2], "b.")
    plt.plot(df.iloc[groups[2], 1], df.iloc[groups[2], 2], "g.")
    plt.subplot(235)
    plt.plot(df.iloc[groups[0], 1], df.iloc[groups[0], 3], "r.")
    plt.plot(df.iloc[groups[1], 1], df.iloc[groups[1], 3], "b.")
    plt.plot(df.iloc[groups[2], 1], df.iloc[groups[2], 3], "g.")
    plt.subplot(236)
    plt.plot(df.iloc[groups[0], 2], df.iloc[groups[0], 3], "r.")
    plt.plot(df.iloc[groups[1], 2], df.iloc[groups[1], 3], "b.")
    plt.plot(df.iloc[groups[2], 2], df.iloc[groups[2], 3], "g.")
    plt.show()
    
    plt.figure()
    plt.subplot(231)
    plt.plot(df.iloc[real_groups[0], 0], df.iloc[real_groups[0], 1], "r.")
    plt.plot(df.iloc[real_groups[1], 0], df.iloc[real_groups[1], 1], "b.")
    plt.plot(df.iloc[real_groups[2], 0], df.iloc[real_groups[2], 1], "g.")
    plt.subplot(232)
    plt.plot(df.iloc[real_groups[0], 0], df.iloc[real_groups[0], 2], "r.")
    plt.plot(df.iloc[real_groups[1], 0], df.iloc[real_groups[1], 2], "b.")
    plt.plot(df.iloc[real_groups[2], 0], df.iloc[real_groups[2], 2], "g.")
    plt.subplot(233)
    plt.plot(df.iloc[real_groups[0], 0], df.iloc[real_groups[0], 3], "r.")
    plt.plot(df.iloc[real_groups[1], 0], df.iloc[real_groups[1], 3], "b.")
    plt.plot(df.iloc[real_groups[2], 0], df.iloc[real_groups[2], 3], "g.")
    plt.subplot(234)
    plt.plot(df.iloc[real_groups[0], 1], df.iloc[real_groups[0], 2], "r.")
    plt.plot(df.iloc[real_groups[1], 1], df.iloc[real_groups[1], 2], "b.")
    plt.plot(df.iloc[real_groups[2], 1], df.iloc[real_groups[2], 2], "g.")
    plt.subplot(235)
    plt.plot(df.iloc[real_groups[0], 1], df.iloc[real_groups[0], 3], "r.")
    plt.plot(df.iloc[real_groups[1], 1], df.iloc[real_groups[1], 3], "b.")
    plt.plot(df.iloc[real_groups[2], 1], df.iloc[real_groups[2], 3], "g.")
    plt.subplot(236)
    plt.plot(df.iloc[real_groups[0], 2], df.iloc[real_groups[0], 3], "r.")
    plt.plot(df.iloc[real_groups[1], 2], df.iloc[real_groups[1], 3], "b.")
    plt.plot(df.iloc[real_groups[2], 2], df.iloc[real_groups[2], 3], "g.")
    plt.show()
    
    # decide proper k, elbow-point method
    errors = []
    for k in xrange(1, 9):
        centroids, groups, err = kmeans(df.iloc[:, :-1], k)
        errors.append(err[-1])
    plt.figure()
    plt.plot(xrange(1, 9), errors, ".-")
    plt.show()
    
