from sklearn import cluster
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np


def main():
    data = datasets.load_iris().data
    EM = [i for i in range(1, 11)]
    y = [np.sum(np.sum((data - np.mean(data, axis=0))**2))]

    for k in EM[1:]:
        kmeans = cluster.k_means(data, n_clusters=k, random_state=0, init='k-means++')
        y.append(kmeans[2])

    plt.plot(EM, y, color='purple')
    plt.xticks(np.arange(1, 11))
    plt.xlabel("K")
    plt.ylabel("Inertia")
    plt.title("Elbow Method for selection of optimal \"K\" clusters")
    plt.annotate("Elbow Point", xy=(2.4, 160), xycoords='data', xytext=(4, 300), textcoords='data',
                 arrowprops=dict(arrowstyle="->", linestyle="--",
                                 connectionstyle="arc3,rad=-0.2",
                                 fc="w"), )
    plt.plot(2, 150, 'o', ms=20, mec='black', mfc='none', mew=2, linestyle="--")
    plt.savefig("elbow.png")




if __name__ == '__main__':
    main()
