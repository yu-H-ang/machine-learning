import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def kmeans(df, nk, niter = 100, tol = 1e-6):
    
    # size of data
    n, d = df.shape[0], df.shape[1]
    
    
    mu = np.zeros(nk, 1)
    sigma = np.zeros(d, d, nk)
    pi = np.zeros(nk, 1)
    
    
    
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
    df = pd.read_csv("./oldfaithful.csv")
    
    df.iloc[:, 0] -= np.mean(df.iloc[:, 0])
    df.iloc[:, 0] /= np.std(df.iloc[:, 0])
    df.iloc[:, 1] -= np.mean(df.iloc[:, 1])
    df.iloc[:, 1] /= np.std(df.iloc[:, 1])
    
    #plt.plot(df.iloc[:, 0], df.iloc[:, 1], '.')
    #plt.axis([-2, 2, -2, 2])
    #plt.show()
    from scipy.interpolate import griddata
    from matplotlib import cm
    
    def plot_countour(x,y,z):
        # define grid.
        xi = np.linspace(-2.1, 2.1, 100)
        yi = np.linspace(-2.1, 2.1, 100)
        ## grid the data.
        zi = griddata((x, y), z, (xi[None,:], yi[:,None]), method='cubic')
        levels = [0.2]
        
        # contour the gridded data, plotting dots at the randomly spaced data points.
        CS = plt.contour(xi,yi,zi,len(levels),linewidths=0.5,colors='k', levels=levels)
        #CS = plt.contourf(xi,yi,zi,15,cmap=plt.cm.jet)
        #CS = plt.contourf(xi,yi,zi,len(levels),cmap=cm.Greys_r, levels=levels)
        #plt.colorbar() # draw colorbar
        # plot data points.
        # plt.scatter(x, y, marker='o', c='b', s=5)
        plt.xlim(-2, 2)
        plt.ylim(-2, 2)
        plt.title('griddata test (%d points)' % npts)
        plt.show()

    def gauss(x,y,Sigma,mu):
        X=np.vstack((x,y)).T
        mat_multi=np.dot((X-mu[None,...]).dot(np.linalg.inv(Sigma)),(X-mu[None,...]).T)
        return  np.diag(np.exp(-1*(mat_multi)))
    
    
    # make up some randomly distributed data
    np.random.seed(1234)
    npts = 1000
    x = np.random.uniform(-2, 2, npts)
    y = np.random.uniform(-2, 2, npts)
    z = gauss(x, y, Sigma=np.asarray([[1.,.5],[0.5,1.]]), mu=np.asarray([0.,0.]))
    plot_countour(x, y, z)
    
