#include "update_centroids.h"

__global__ void
update_centroids_init(DATATYPE* points, DATATYPE* centroids, int* labels, int* count_labels, int K, int dim)
{
	int i;
	for(i = tidu; i < K*dim; i+=BLOCK_SIZE_UPDATE*N_BLOCKS_UPDATE){
		centroids[i] = 0;
	}
}

__global__ void
update_centroids_divide(DATATYPE* points, DATATYPE* centroids, int* labels, int* count_labels, int K, int dim)
{
	int i;
	for(i = tidu; i < K*dim; i+=BLOCK_SIZE_UPDATE*N_BLOCKS_UPDATE){
		if(count_labels[i/dim] > 0)
			centroids[i] = centroids[i]/count_labels[i/dim];
	}
}

__global__ void
update_centroids_sum(DATATYPE* points, DATATYPE* centroids, int* labels, int* count_labels, int K, int N, int dim)
{
	#if UPDATE_CENTROID == UPDATE_ON_GPU

	int i,k;

	
	#ifdef PRIVATE_ATOMIC_UPDATE
		//Inicializa os centroides locais de cada bloco
		__shared__ DATATYPE sm_centroids[MAX_DIM*MAX_K];
		for(i = tidx; i < dim*K; i+=BLOCK_SIZE_UPDATE){
			sm_centroids[i] = 0;
		}
		__syncthreads();
	#endif
	
	
	for(i=tidu; i < N*dim; i+=BLOCK_SIZE_UPDATE*N_BLOCKS_UPDATE){
		
		for(k = 0; k < K; ++k){
			if(labels[i/dim] == k){
				#ifdef PRIVATE_ATOMIC_UPDATE
					atomicAdd(&sm_centroids[k*dim+(i%dim)], points[i]);
				#else
					atomicAdd(&centroids[k*dim+(i%dim)], points[i]);
				#endif
				
			}
		}
	}

	#ifdef PRIVATE_ATOMIC_UPDATE
		//Cada bloco coloca o valor parcial nos centroides da memoria global
		//Apos esse kernel, havera outra funcao para dividir os valores dos elementos
		__syncthreads();
		for(i = tidx; i < K*dim; i+=BLOCK_SIZE_UPDATE){
			atomicAdd(&centroids[i], sm_centroids[i]);
		}
	#endif

	#endif
}