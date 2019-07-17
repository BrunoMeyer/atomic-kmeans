#include <stdlib.h>
#include <stdio.h>

int* double_matrix(int* input, int w, int h){
	// fprintf(stderr,"aa %d\n",input[0]);
	int* output = malloc(sizeof(int)*w*h);
	for(int i=0; i < h; ++i){
		for(int j=0; j < w; ++j){
			output[i*w+j]= input[i*w+j]*5;
		}
	}

	return output;
}
