#ifndef UPDATECENTROID_H
#define UPDATECENTROID_H


// #include <stdio.h>
// #include <stdlib.h>

// #include <string>
// #include <vector>
// #include <sstream> //istringstream
// #include <iostream> // cout
// #include <fstream> // ifstream
// #include "clock.h"

#include "kmeans_config.h"

__global__ void update_centroids_init(float* points, float* centroids, int* labels, int* count_labels, int K, int dim);

__global__ void update_centroids_divide(float* points, float* centroids, int* labels, int* count_labels, int K, int dim);

__global__ void update_centroids_sum(float* points, float* centroids, int* labels, int* count_labels, int K, int N, int dim);

#endif