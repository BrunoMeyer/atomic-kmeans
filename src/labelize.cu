#include "labelize.h"

__global__ void
init_count_labels(DATATYPE* points, DATATYPE* centroids, int* labels, int* count_labels, int K)
{
	for(int i = tid; i < K; i+=BLOCK_SIZE*N_BLOCKS){
		count_labels[i] = 0;
	}
}

__global__ void
create_labels(DATATYPE* points, DATATYPE* centroids, int* labels, int* count_labels, int K, int N, int dim)
{
	
	int i,j,k, smaller_label;
	DATATYPE smaller, aux, diff;
	
	// Inicializa a contagem de elementos relacionados a cada centroide
	// Ao final, cada bloco colocara esses valores na global memory
	#ifdef PRIVATE_ATOMIC_LABEL
		__shared__ int smem_count_labels[MAX_K];
		for(i = tidx; i < K; i+=BLOCK_SIZE){
			smem_count_labels[tidx] = 0;
		}
		__syncthreads();
	#endif
	
	

	// Separa um ponto para cada thread e trata o resto
	// Como thread persistente
	for(i = tid; i < N; i+=BLOCK_SIZE*N_BLOCKS){
		
		smaller = 0;
		smaller_label = 0;

		// Trata o primeiro centroide individualmente
		for(j = 0; j < dim; ++j){
			diff = (centroids[j] - points[i*dim+j]);
			smaller+=  diff*diff;
		}

		// Calcula a distancia para os outros centroides
		for(k = 1; k < K; ++k){
			aux = 0;
			for(j = 0; j < dim; ++j){
				diff = (centroids[j + k*dim] - points[i*dim+j]);
				aux+= diff*diff;
			}
			if(aux < smaller){
				smaller = aux;
				smaller_label = k;
			}
		}
		
		// Acesso a global memory (sabemos que nao havera concorrencia de acesso)
		// O acesso parece estar coalest
		labels[i] = smaller_label;
		#ifdef PRIVATE_ATOMIC_LABEL
			atomicAdd(&smem_count_labels[smaller_label], 1);
		#else
			atomicAdd(&count_labels[smaller_label], 1);
		#endif
	}

	#ifdef PRIVATE_ATOMIC_LABEL
		__syncthreads();
		for(i = tidx; i < K; i+=BLOCK_SIZE){
			atomicAdd(&count_labels[tidx], smem_count_labels[tidx]);
		}
	#endif
}