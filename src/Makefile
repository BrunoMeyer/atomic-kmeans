kmeans: labelize.o update_centroids.o
	# nvcc -O3 -D DATASET=FAKE_DATASET -D UPDATE_CENTROID=UPDATE_ON_GPU kmeans.cu -o kmeans labelize.o update_centroids.o
	nvcc -O3 --compiler-options -fPIC -c kmeans.cu labelize.cu update_centroids.cu
	nvcc -O3 -shared -o kmeans.so kmeans.o labelize.o update_centroids.o

labelize.o: labelize.cu
	nvcc -O3 -c labelize.cu -o labelize.o

update_centroids.o: update_centroids.cu
	nvcc -O3 -c update_centroids.cu -o update_centroids.o

clean:
	rm kmeans.o labelize.o update_centroids.o
	rm kmeans.so