extern "C" {
#include "kmeans.h"
}

#include <stdio.h>
#include <stdlib.h>

#include <string>
#include <vector>
#include <sstream> //istringstream
#include <iostream> // cout
#include <fstream> // ifstream
#include "clock.h"


#include "kmeans_config.h"
#include "labelize.h"
#include "update_centroids.h"



using namespace std;


#if DATASET == REAL_DATASET
int load_data_from_csv(DATATYPE* points){
	
	int i,j;
	i = 0;
	int err = 0;
	FILE* file = fopen(DATASET_NAME, "r");
	//Carrega as informacoes da base de treinamento para a memoria
	while (!feof (file)){
		for(j = 0; j < DIMENSION-1; ++j){
			err = fscanf(file, "%f,", &points[i*DIMENSION+j]);
        }
        err = fscanf(file, "%f\n", &points[i*DIMENSION+j]);
		i++;
	}
	fclose (file);
	return err;
}
#elif DATASET == FAKE_DATASET
	void initialize_fake_data(DATATYPE* points){
		srand(RANDOM_SEED);
		int i,j;
		for(i = 0; i < N_POINTS; ++i){
			
			for(j = 0; j < DIMENSION; ++j){
				int aux = (i/(ELEMENTS_PER_CLUSTER));
				DATATYPE r = rand() % DISTANCE_INTRA_CLUSTER;
				points[i*DIMENSION+j] = aux*DISTANCE_EXTRA_CLUSTER + r-DISTANCE_INTRA_CLUSTER/2;
			}
		}
	}
#endif


void log_centroids(DATATYPE* centroids, int K, int dim){
	int k,j;
	for(k = 0; k < K; ++k){
		printf("Centroide %d:\n", k);
		for(j = 0; j < dim; ++j){
			printf("\t%f\n", centroids[k*dim+j]);
		}
		printf("\n");
		printf("\n");
	}
}
void log(DATATYPE* points, int* labels, int N, int dim){
	int i,j;

	// printf("----------\n");

	for(i = 0; i < N; ++i){
		// printf("%d\t", labels[i]);
		// printf("{%d}:\t\t", labels[i]);
		for(j = 0; j < dim; ++j){
			printf("%f\t", points[i*dim+j]);
		}
		printf("\n");
	}
	// printf("\n");
	// printf("----------\n");
}



#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

int* kmeans(int K, DATATYPE* points, int N, int dim, int max_iter, int verbose)
{
	chronometer_t* chr_global = (chronometer_t*)malloc(sizeof(chronometer_t));
	chrono_reset(chr_global);
	chrono_start(chr_global);

	DATATYPE* centroids = (DATATYPE*) malloc(sizeof(DATATYPE)*K*dim);
	int* labels = (int*)malloc(sizeof(int)*N);
	int* count_labels = (int*)malloc(sizeof(int)*K);
	
	int i,j,k;
	
	// log(points, labels, N, dim);
	
	for(k = 0; k < K; ++k){
		for(j = 0; j < dim; ++j){
			// centroids[k*DIMENSION+j] = points[(N_POINTS/(k+1))-1+j];
			centroids[k*dim+j] = points[(k*N/K)*dim + j];
			// fprintf(stderr,"%lf\t",centroids[k*dim+j]);
		}
		// fprintf(stderr,"\n");
	}

	// log_centroids(centroids, K, dim);

	DATATYPE* d_points;
	DATATYPE* d_centroids;
	int* d_labels;
	int* d_count_labels;
	
	gpuErrchk(cudaMalloc((void **)&d_points, N*dim*sizeof(DATATYPE)));
	gpuErrchk(cudaMalloc((void **)&d_centroids, K*dim*sizeof(DATATYPE)));
	gpuErrchk(cudaMalloc((void **)&d_labels, N*sizeof(int)));
	gpuErrchk(cudaMalloc((void **)&d_count_labels, K*sizeof(int)));

	chronometer_t* chr_transfer = (chronometer_t*)malloc(sizeof(chronometer_t));
	chrono_reset(chr_transfer);

	chrono_start(chr_transfer);
	cudaMemcpy(d_points, points, N*dim*sizeof(DATATYPE), cudaMemcpyHostToDevice);
	cudaMemcpy(d_centroids, centroids, sizeof(DATATYPE)*K*dim, cudaMemcpyHostToDevice);
	// cudaMemcpy(d_centroids, centroids, K*DIMENSION*sizeof(DATATYPE));
	// cudaMemcpy(d_labels, labels, N_POINTS*DIMENSION*sizeof(DATATYPE), cudaMemcpyHostToDevice);
	// cudaMemcpy(d_count_labels, points, N_POINTS*DIMENSION*sizeof(DATATYPE), cudaMemcpyHostToDevice);
	chrono_stop(chr_transfer);


	int threadsPerBlock = BLOCK_SIZE;
	
	int blocksPerGrid = N_BLOCKS;//(N_POINTS + threadsPerBlock - 1) / threadsPerBlock;

	
	chronometer_t* chr_loop = (chronometer_t*)malloc(sizeof(chronometer_t));
	chronometer_t* chr_total = (chronometer_t*)malloc(sizeof(chronometer_t));
	chronometer_t* chr_label = (chronometer_t*)malloc(sizeof(chronometer_t));
	chronometer_t* chr_update = (chronometer_t*)malloc(sizeof(chronometer_t));
	
	chrono_reset(chr_loop);
	chrono_reset(chr_total);
	chrono_reset(chr_label);
	chrono_reset(chr_update);

	long long unsigned min_total = 0;
	long long unsigned min_label = 0;
	long long unsigned min_update = 0;
	int first = 1;
	
	
	
	chrono_reset(chr_loop);  
	chrono_start(chr_loop);


	for(int ite = 0; ite < max_iter; ++ite){
		if(verbose>=1){
			printf("\rIterations of KMeans: |");
			for(i=0; i< 1+(int)( ((float)(ite)/max_iter) *80); ++i){
				printf("#");
			}
			for(i=i; i<80;++i){
				printf(".");
			}
			printf("| %d %%",1+(int)(100*((float)(ite)/max_iter)));
			fflush(stdout);
		}

		// chrono_reset(chr_total); 
    	chrono_start(chr_total);


		// Rotulacao
		////////////////////////////////////////////////////////////////////////
		// chrono_reset(chr_label);
    	chrono_start(chr_label);
		// cudaDeviceSynchronize();
		// Inicializa em um kernel diferente
		// Eh necessario pois nao eh trivial sincronizar os blocos
		init_count_labels<<<blocksPerGrid, threadsPerBlock>>>(
			d_points, d_centroids, d_labels, d_count_labels, K
		);
		
		create_labels<<<blocksPerGrid, threadsPerBlock>>>(
			d_points, d_centroids, d_labels, d_count_labels, K, N, dim
		);
		
		cudaDeviceSynchronize();
		chrono_stop(chr_label);
		if(first || chr_label->xtotal_ns < min_label){
			min_label = chr_label->xtotal_ns;
		}
		////////////////////////////////////////////////////////////////////////



		// chrono_reset(chr_update);
		chrono_start(chr_update);
		// Atualiza em CPU ou GPU
		#if UPDATE_CENTROID == UPDATE_ON_GPU
			update_centroids_init<<<N_BLOCKS_UPDATE,BLOCK_SIZE_UPDATE>>>(
				d_points, d_centroids, d_labels, d_count_labels, K, dim
			);
			// cudaDeviceSynchronize();
			update_centroids_sum<<<N_BLOCKS_UPDATE,BLOCK_SIZE_UPDATE>>>(
				d_points, d_centroids, d_labels, d_count_labels, K, N, dim
			);
			// cudaDeviceSynchronize();
			// printf("\n\n");
			update_centroids_divide<<<N_BLOCKS_UPDATE,BLOCK_SIZE_UPDATE>>>(
				d_points, d_centroids, d_labels, d_count_labels, K, dim
			);
			cudaDeviceSynchronize();
		#elif UPDATE_CENTROID == UPDATE_ON_CPU
			chrono_start(chr_transfer);
			cudaMemcpy(labels, d_labels, N*sizeof(int),cudaMemcpyDeviceToHost);
			cudaMemcpy(count_labels, d_count_labels, K*sizeof(int),cudaMemcpyDeviceToHost);
			chrono_stop(chr_transfer);
			
			// puts("##");
			// for(j=0; j < K;++j){
			// 	printf("%d\n",count_labels[j]);
			// }
			// printf("\n\n");
			// for(j=0; j < N; ++j){
			// 	printf("%d\n",labels[j]);
			// }
			// puts("##");

			// cudaDeviceSynchronize();


			for(k = 0; k < K; ++k){
				for(j = 0; j < dim; ++j){
					centroids[k*dim+j] = 0;
				}
			}
	
			for(i = 0; i < N; ++i){
				
				for(k = 0; k < K; ++k){
					if(labels[i] == k){
						for(j = 0; j < dim; ++j){
							centroids[k*dim+j] += points[i*dim+j];
						}
					}
				}
			}
	
			for(k = 0; k < K; ++k){
				for(j = 0; j < dim; ++j){
					if(count_labels[k] > 0) centroids[k*dim+j] /= count_labels[k];
				}
			}

			chrono_start(chr_transfer);
			cudaMemcpy(d_centroids, centroids, sizeof(DATATYPE)*K*dim, cudaMemcpyHostToDevice);
			chrono_stop(chr_transfer);
		#endif
		chrono_stop(chr_update);
		if(first || chr_update->xtotal_ns < min_update){
			min_update = chr_update->xtotal_ns;
		}
		////////////////////////////////////////////////////////////////////////

		chrono_stop(chr_total);
		if(first || chr_total->xtotal_ns < min_total){
			min_total = chr_total->xtotal_ns;
		}
		first = 0;
		
		// log_centroids(centroids, K, dim);
	}
	chrono_stop(chr_loop);
	
	if(verbose>=1){
		printf("\rIterations of KMeans: |");
		for(i=0; i< 80; ++i){
			printf("#");
		}
		printf("| 100 %%\n");
		fflush(stdout);
	}
	

	if(verbose>=2){
		// printf("Labeling on GPU takes %lf seconds\n",min_label/(float)10e8);
		// printf("Updating centroids on GPU takes %lf seconds\n",min_update/(float)10e8);
		// printf("Total KMEANS on GPU takes %lf seconds\n",min_total/(float)10e8);
		// printf("All iterations KMEANS on GPU takes %lf seconds\n",chr_loop->xtotal_ns/(float)10e8);
		
		printf("Labeling on GPU takes %lf seconds\n",chr_label->xtotal_ns/(float)10e8);
		printf("Updating centroids on GPU takes %lf seconds\n",chr_update->xtotal_ns/(float)10e8);
		printf("Total KMEANS on GPU takes %lf seconds\n",chr_total->xtotal_ns/(float)10e8);
		printf("All iterations KMEANS on GPU takes %lf seconds\n",chr_loop->xtotal_ns/(float)10e8);
	}
	
	chrono_start(chr_transfer);
	// cudaMemcpy(points, d_points, N*dim*sizeof(DATATYPE),cudaMemcpyDeviceToHost);
	// cudaMemcpy(centroids, d_centroids, sizeof(DATATYPE)*K*dim,cudaMemcpyDeviceToHost);
	cudaMemcpy(labels, d_labels, N*sizeof(int),cudaMemcpyDeviceToHost);
	chrono_stop(chr_transfer);
	cudaDeviceSynchronize();
	printf("Transfer CPU<->GPU KMEANS takes %lf seconds\n",chr_transfer->xtotal_ns/(float)10e8);
	
	// log(points,labels);
	// log_centroids(centroids, K, dim);
	
	cudaDeviceSynchronize();
	cudaFree(d_centroids);
	cudaFree(d_count_labels);
	cudaFree(d_labels);
	cudaFree(d_points);
	// free(points);
	// free(centroids);
	// free(count_labels);
	// free(labels);

	chrono_stop(chr_global);
	if(verbose>=2){
		printf("Global KMEANS takes %lf seconds\n",chr_global->xtotal_ns/(float)10e8);
	}
	return labels;
}

int* double_matrix(int* input, int w, int h){
	// fprintf(stderr,"aa %d\n",input[0]);
	int* output = (int*)malloc(sizeof(int)*w*h);
	for(int i=0; i < h; ++i){
		for(int j=0; j < w; ++j){
			output[i*w+j]= input[i*w+j]*5;
		}
	}

	return output;
}

// int main(int argc, char *argv[]){
// 	int K = atoi(argv[1]);
// 	DATATYPE* points = (DATATYPE*)malloc(sizeof(DATATYPE)*N_POINTS*DIMENSION);

// 	#if DATASET == REAL_DATASET
// 		load_data_from_csv(points);
// 	#elif DATASET == FAKE_DATASET
// 		initialize_fake_data(points);
// 	#endif

	

// 	int i,j;
// 	for(i=0; i < N_POINTS; ++i){
// 		for(j=0; j < DIMENSION; ++j){
// 			// points[i*DIMENSION+j] = 100*((i) / (N_POINTS/NUMBER_FAKE_CLUSTER));
// 			printf("%d: %lf\t", i, points[i*DIMENSION+j]);
// 		}
// 		printf("\n");
// 	}
	
// 	int* labels = kmeans(K,points,N_POINTS,DIMENSION);

// 	for(i=0; i < N_POINTS; ++i){
// 		printf("%d: %d\n", i, labels[i]);
// 	}

	
	
// 	return 0;
// }
