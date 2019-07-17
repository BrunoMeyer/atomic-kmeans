from ctypes import cdll
from ctypes import c_char_p
import ctypes
import numpy as np

from numpy.ctypeslib import ndpointer


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




kmeans_lib = ctypes.CDLL("kmeans.so")

kmeans_f = kmeans_lib.kmeans




def kmeans(K, m):
    # WARNING: Overwrite variables may cause problems with garbage collector
    m2 = np.array(m, dtype=ctypes.c_float)
    m_shape = m.shape
    h,w = m_shape

    m_type = ndpointer(dtype=ctypes.c_float, shape=m_shape)
    h = ctypes.c_int(h)
    w = ctypes.c_int(w)
    k = ctypes.c_int(K)

    m3 = m2.ctypes.data_as(m_type)

    kmeans_f.restype = ndpointer(dtype=ctypes.c_int, shape=(m_shape[0]))
    x = kmeans_f(k,m3,h,w)
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

from sklearn import datasets
iris = datasets.load_iris()
# X = iris.data[:, :2]  # we only take the first two features.
X = iris.data
y = iris.target


labels = kmeans(3,X)
[print(l1,l2) for l1,l2 in zip(labels,y)]