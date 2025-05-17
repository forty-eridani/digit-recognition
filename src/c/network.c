#include "network.h"

#include <assert.h>
#include <stdlib.h>
#include <time.h>

static void fillArrayRandom(double arr[], int count) {
	srand(time(NULL));

	for (int i = 0; i < count; i++) {
		arr[i] = (double)rand() / RAND_MAX;
	}
}

Vector CalculateVectorCost(Vector observed, Vector expected) {
	Vector r = SubtractVectors(observed, expected);
	MultiplyVectorsInPlace(r, &r);

	return r;
}

Network CreateNetwork(int* layers, int layerCount) {
	// The network is represented by a bunch of adjacency matrices 
	// Each column represents the layer ahead, and each row represents
	// the neuron that it leads to. Each cell in the matrix represents a
	// weight.
	
	Network network = {.neuronLayerCount = layerCount - 1, 
					   .layers = malloc((layerCount - 1) * sizeof(Matrix)),
	                   .biases = malloc((layerCount - 1) * sizeof(Vector))};

	
	for (int i = 0; i < layerCount - 1; i++) {
		int matrixSize = layers[i] * layers[i + 1];

		double matrixArray[matrixSize];
		fillArrayRandom(matrixArray, matrixSize);

		double biases[layers[i + 1]];
		fillArrayRandom(biases, layers[i + 1]);

		network.layers[i] = CreateMatrix(matrixArray, layers[i + 1], layers[i]);
		network.biases[i] = CreateVector(biases, layers[i + 1]);
	}

	return network;
}
