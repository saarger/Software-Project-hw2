import random
import numpy as np
import pandas as pd
import mykmeanssp
import sys

def d(p,q):
    sum = 0
    for i in range(len(p)):
        sum += (p[i]-q[i])**2
    return sum**0.5

def arg_parsing():
    try:

        args = sys.argv
        #check length of args and if the input is valid
       
        if len(args)>6 or len(args)<5:
            raise Exception
        k = int(args[1])
        epsilon = float(args[-3])
        if len(args) == 5:
            max_iter = 300
            idx_of_file = 3
        else:
            idx_of_file = 4
            max_iter = int(args[2])

        input1_filename = args[idx_of_file]
        input2_filename = args[idx_of_file + 1]

        if max_iter<=1 or max_iter>=1000:
            print("Invalid maximum iteration!")
            sys.exit(1)
        run_kmeans_pp(k, max_iter,epsilon, input1_filename, input2_filename)
    except Exception:
        print("An Error Has Occurred")
        sys.exit(1)


def kmeans_pp(k,vectors):

    np.random.seed(0)
    centroids = []
    indexes = []
    n_array = np.arange(len(vectors))
    j = np.random.choice(n_array)
    centroids.append( vectors[j])
    indexes.append(j)
    
    for i in range(1,k):
        distances = []
        for vector in vectors:
            min_distance = min(d(vector, centroid) for centroid in centroids)
            distances.append(min_distance)
        distances = np.array(distances)
        sum = np.sum(distances)
        probabilities = distances/sum
        j = np.random.choice(n_array, p=probabilities)
        centroids.append(vectors[j])
        indexes.append(j)
        

    s = ""
    for i in indexes:
        s += str(i) + ","
    print(s[:-1])
    
    return centroids

def read_files(input1_filename, input2_filename):
    df1 = pd.read_csv(input1_filename,header=None)
    df2 = pd.read_csv(input2_filename,header=None)
    df = pd.merge(df1, df2, on=0, how='inner')
    df = df.sort_values(by=0)
    return df.iloc[:, 1:].values.tolist()


def run_kmeans_pp(k, max_iter,epsilon, input1_filename, input2_filename):
    vectors = read_files(input1_filename, input2_filename)
    
    init_centroids = kmeans_pp(k,vectors)
    centroids = mykmeanssp.fit(k, max_iter, epsilon, vectors, init_centroids, len(vectors), len(vectors[0]))
    for centroid in centroids:
        for i, element in enumerate(centroid):
            if i != 0:
                print(",", end="")
            print(f"{element:.4f}", end="")
        print()

if __name__ == '__main__':

    
    arg_parsing()
    


