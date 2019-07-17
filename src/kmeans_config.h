#ifndef KMEANSCONFIG_H
#define KMEANSCONFIG_H

#include <stdio.h>
#include <stdlib.h>

// #include <string>
// #include <vector>
// #include <sstream> //istringstream
// #include <iostream> // cout
// #include <fstream> // ifstream
// #include "clock.h"



// #define PRIVATE_ATOMIC 1

#define UPDATE_ON_CPU 1
#define UPDATE_ON_GPU 2

#ifndef UPDATE_CENTROID
	#define UPDATE_CENTROID UPDATE_ON_GPU
#endif
// #define UPDATE_CENTROID UPDATE_ON_GPU


#define REAL_DATASET 1
#define FAKE_DATASET 2


//Blocos utilizados na rotulacao
#ifndef N_BLOCKS
	#define N_BLOCKS 1
#endif
#define BLOCK_SIZE 1024


//Blocos utilizados na atualizacao do centroide
// #define N_BLOCKS_UPDATE N_BLOCKS
#define N_BLOCKS_UPDATE N_BLOCKS
#define BLOCK_SIZE_UPDATE 1024

#define bid blockIdx.x
#define tid (BLOCK_SIZE * blockIdx.x + threadIdx.x)
#define tidu (BLOCK_SIZE_UPDATE * blockIdx.x + threadIdx.x)
#define tidx threadIdx.x

#define MAX_K 512
#define MAX_DIM 1000
//Maximo de iteracoes
#define MAX_ITE 100

#ifndef DATASET
	#define DATASET FAKE_DATASET
#endif

#if DATASET == REAL_DATASET
	#define N_POINTS 5820
	#define DIMENSION 33
	#define DATASET_NAME "dataset.csv"
#elif DATASET == FAKE_DATASET
	#define RANDOM_SEED 1
	#define N_POINTS 30
	#define DIMENSION 3
	//Parametros utilzados para sintetizar os dados 
	#define NUMBER_FAKE_CLUSTER 3
	#define ELEMENTS_PER_CLUSTER (N_POINTS/NUMBER_FAKE_CLUSTER)
	#define DISTANCE_INTRA_CLUSTER 50
	#define DISTANCE_EXTRA_CLUSTER 200
#endif


#endif
