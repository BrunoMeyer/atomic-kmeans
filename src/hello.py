from ctypes import cdll
from ctypes import c_char_p
import ctypes
import numpy as np

from numpy.ctypeslib import ndpointer
from sklearn import datasets

from sklearn.cluster import KMeans
from sklearn.utils.linear_assignment_ import linear_assignment
import matplotlib.pyplot as plt

from sklearn.utils import shuffle

import time

from sklearn import metrics
from sklearn import preprocessing

import scipy.sparse as sps


RANDOM_SEED = 0
'''
hello_lib = ctypes.CDLL("kmeans.so")
# hello_lib = ctypes.CDLL("hello.so")

double_matrix = hello_lib.double_matrix


m = np.array([[1,2],[3,4]], dtype=ctypes.c_int)
m_shape = m.shape
h,w = m_shape

m_type = ndpointer(dtype=ctypes.c_int, shape=m.shape)
h = ctypes.c_int(h)
w = ctypes.c_int(w)


m = m.ctypes.data_as(m_type)

double_matrix.restype = ndpointer(dtype=ctypes.c_int, shape=m_shape)
x = double_matrix(m,w,h)
# print(np.ctypeslib.as_array(res, shape=res_shape))
print(np.ctypeslib.as_array(x, shape=m_shape))
'''



kmeans_lib = ctypes.CDLL("kmeans.so")

kmeans_f = kmeans_lib.kmeans

def _make_cost_m(cm):
    s = np.max(cm)
    return (- cm + s)

def compare_results(X,labels,real_labels, total_classes, plot=False):
    
    # result = np.zeros((total_classes,total_classes))
    # for l1,l2 in zip(labels,real_labels):
    #     result[l1,l2]+=1

    # result = sps_acc = sps.coo_matrix((total_classes,total_classes))
    # for l1,l2 in zip(labels,real_labels):
    #     result[l1,l2]+=1

    # indexes = linear_assignment(_make_cost_m(result))
    # js = [e[1] for e in sorted(indexes, key=lambda x: x[0])]
    # result2 = result[:, js]


    # if plot:
    #     plt.imshow(result2, cmap='seismic', interpolation='nearest')
    #     plt.show()

    # acc = np.trace(result2)/np.sum(result2)
    # print("Accuracy: {}".format(acc))
    
    # sl = metrics.silhouette_score(X, labels, metric='euclidean')
    # print("Silhouette: {}".format(sl))
    nmis = metrics.normalized_mutual_info_score(real_labels, labels, average_method='arithmetic')
    print("NMIS: {}".format(nmis))



def kmeans(K, m, max_iter=300, verbose=2, dtype=ctypes.c_float):
    # WARNING: Overwrite variables may cause problems with garbage collector
    if verbose>=2:
        print("\nConverting data...")
    
    m2 = np.array(m, dtype=dtype)
    m_shape = m.shape
    h,w = m_shape
    
    m_type = ndpointer(dtype=dtype, shape=m_shape)
    h = ctypes.c_int(h)
    w = ctypes.c_int(w)
    k = ctypes.c_int(K)
    mi = ctypes.c_int(max_iter)
    verb = ctypes.c_int(verbose)

    m3 = m2.ctypes.data_as(m_type)

    kmeans_f.restype = ndpointer(dtype=ctypes.c_int, shape=(m_shape[0]))
    
    if verbose>=2:
        print("Calling C++/CUDA function...")

    start_time = time.time()
    x = kmeans_f(k,m3,h,w,mi,verb)
    print("\t--- %s seconds ---" % (time.time() - start_time))
    # print(np.ctypeslib.as_array(res, shape=(m_shape[0]) )

    if verbose>=2:
        print("Converting results...\n")
    x = np.ctypeslib.as_array(x, shape=(m_shape[0]))
    return x

dataset = []

# N_POINTS = 30
# DIM = 3
# CLUSTERS = 3
# DIS_CLUSTER = 100
# for i in range(N_POINTS):
#     dataset.append([DIS_CLUSTER*(int(i*CLUSTERS/N_POINTS))+np.random.random() for j in range(DIM)])



# labels = kmeans(2,dataset)
# print(labels)
# labels = kmeans(3,dataset)
# print(labels)


# db = datasets.load_iris()
# db = datasets.load_digits()
# db = datasets.fetch_lfw_people(min_faces_per_person=None, resize=0.4, data_home="/tmp/")
# db = datasets.fetch_lfw_people(resize=0.4, data_home="/tmp/")
# db = datasets.fetch_lfw_people(data_home="/tmp/")

import tensorflow as tf

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path='/tmp/mnist.npz')
x_train = x_train.reshape((x_train.shape[0],-1))
class DB(object):
    def __init__(self,data,target):
        self.data = data
        self.target = target

db = DB(x_train,y_train)

for i in range(10):
    print("\n"*3)
    print("#"*80)
    # X = db.data[:, :2]  # we only take the first two features.
    X = db.data
    y = db.target
    X, y = shuffle(X, y, random_state=RANDOM_SEED+i)

    N_SAMPLES = 10000000
    N_DIM = 1000000
    X = X[:N_SAMPLES,:N_DIM]
    y = y[:N_SAMPLES]

    le = preprocessing.LabelEncoder()
    y = le.fit_transform(y)
    # min_v = max(np.min(np.abs(X)), 10e-6)
    # X = np.array(X/min_v, dtype=np.float32)
    total_classes = len(set(y))

    # print(total_classes, X.shape)
    # exit()


    visualize = False
    max_iter = 30
    # K = 64
    K = total_classes

    print("Local KMeans")
    start_time = time.time()
    labels = kmeans(K,X, max_iter=max_iter)
    print("--- %s seconds ---" % (time.time() - start_time))
    compare_results(X,labels,y,total_classes,plot=visualize)
    print("\n\n")

    print("Sklearn KMeans")
    start_time = time.time()
    model = KMeans(n_clusters=K, random_state=RANDOM_SEED, max_iter=max_iter,
                    init="random",precompute_distances=False,algorithm="full",
                    verbose=0, tol=-1, n_init=1,n_jobs=1).fit(X)
    # model = KMeans(n_clusters=K, random_state=RANDOM_SEED, verbose=0, max_iter=max_iter).fit(X)
    print("--- %s seconds ---" % (time.time() - start_time))
    labels = model.labels_
    compare_results(X,labels,y,total_classes,plot=visualize)