kmeans: labelize.o update_centroids.o
	# nvcc -O3 -D DATASET=FAKE_DATASET -D UPDATE_CENTROID=UPDATE_ON_GPU kmeans.cu -o kmeans labelize.o update_centroids.o
	nvcc --compiler-options -fPIC -c kmeans.cu labelize.cu update_centroids.cu
	nvcc -shared -o kmeans.so kmeans.o labelize.o update_centroids.o

labelize.o: labelize.cu
	nvcc -c labelize.cu -o labelize.o

update_centroids.o: update_centroids.cu
	nvcc -c update_centroids.cu -o update_centroids.o

clean:
	rm kmeans.o labelize.o update_centroids.o
	rm kmeans.so
	rm kmeans