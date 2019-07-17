#ifndef LABELIZE_H
#define LABELIZE_H

// #include <stdio.h>
// #include <stdlib.h>

// #include <string>
// #include <vector>
// #include <sstream> //istringstream
// #include <iostream> // cout
// #include <fstream> // ifstream
// #include "clock.h"

#include "kmeans_config.h"

__global__ void init_count_labels(float* points, float* centroids, int* labels, int* count_labels, int K);

__global__ void create_labels(float* points, float* centroids, int* labels, int* count_labels, int K, int N, int dim);

#endif