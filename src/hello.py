from ctypes import cdll
from ctypes import c_char_p
import ctypes
import numpy as np

from numpy.ctypeslib import ndpointer
from sklearn import datasets

from sklearn.cluster import KMeans
from sklearn.utils.linear_assignment_ import linear_assignment
import matplotlib.pyplot as plt

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

def compare_results(labels,real_labels, total_classes, plot=False):
    
    result = np.zeros((total_classes,total_classes))
    for l1,l2 in zip(labels,real_labels):
        result[l1][l2]+=1

    indexes = linear_assignment(_make_cost_m(result))
    js = [e[1] for e in sorted(indexes, key=lambda x: x[0])]
    result2 = result[:, js]


    if plot:
        plt.imshow(result2, cmap='seismic', interpolation='nearest')
        plt.show()

    acc = np.trace(result2)/np.sum(result2)
    print("Accuracy: {}".format(acc))


def kmeans(K, m, max_iter=300):
    # WARNING: Overwrite variables may cause problems with garbage collector
    m2 = np.array(m, dtype=ctypes.c_float)
    m_shape = m.shape
    h,w = m_shape

    m_type = ndpointer(dtype=ctypes.c_float, shape=m_shape)
    h = ctypes.c_int(h)
    w = ctypes.c_int(w)
    k = ctypes.c_int(K)
    mi = ctypes.c_int(max_iter)

    m3 = m2.ctypes.data_as(m_type)

    kmeans_f.restype = ndpointer(dtype=ctypes.c_int, shape=(m_shape[0]))
    x = kmeans_f(k,m3,h,w,mi)
    # print(np.ctypeslib.as_array(res, shape=(m_shape[0]) )
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
db = datasets.load_digits()


# X = db.data[:, :2]  # we only take the first two features.
X = db.data
y = db.target
total_classes = len(set(y))


labels = kmeans(total_classes,X, max_iter=100)
compare_results(labels,y,total_classes)


kmeans = KMeans(n_clusters=total_classes, random_state=0, max_iter=100,
                init="random",precompute_distances=False,algorithm="full",
                verbose=0, tol=-1, n_init=1).fit(X)
labels = kmeans.labels_
compare_results(labels,y,total_classes)