#include "network.h"
#include "matrix.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define ERROR_OUTPUT (Vector){.size = 0, .data = NULL}

static void fillArrayRandom(double arr[], int count) {
	srand(time(NULL));

	for (int i = 0; i < count; i++) {
		arr[i] = (double)rand() / RAND_MAX;
	}
}

static double relu(double value) {
	return value > 0 ? value : 0;
}

static double reluPrime(double x) {
	return x > 0 ? 1 : 0;
}

static void softmax(Vector v) {
	for (int i = 0; i < v.size; i++) {
		v.data[i] = exp(v.data[i]);
	}

	double sum = VectorSum(v);

	for (int i = 0; i < v.size; i++) {
		v.data[i] = v.data[i] / sum;
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
	
	assert(layers);
	
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

Vector FeedForward(Network network, Vector input, int currentLayer) {
	assert(input.data);
	assert(network.layers);
	assert(network.biases);

	printf("Input on layer %d\n", currentLayer);
	PrintVector(input);

	// We have traversed through all of the layers
	if (currentLayer == network.neuronLayerCount)
		return input;

	if (input.size != network.layers[currentLayer].cols) {
		printf("Input dimensions do not match that of neural network.\n");
		return ERROR_OUTPUT;	
	}

	int activationSize = network.layers[currentLayer].rows;

	Vector currentActivation = {.size = activationSize,
								.data = malloc(activationSize * sizeof(double))};

	for (int i = 0; i < activationSize; i++) {
		// We do a microplastic amount of memory tomfoolery
		const double* rowAddr = network.layers[currentLayer].data + network.layers[currentLayer].cols * i;

		Vector weightedActivations = CreateVector(rowAddr, input.size);
		MultiplyVectorsInPlace(input, &weightedActivations);

		double z = VectorSum(weightedActivations) + network.biases[currentLayer].data[i];

		// This means we are on the output layer and only want to apply softmax to the z values
		if (currentLayer == network.neuronLayerCount - 1) {
			currentActivation.data[i] = z;
		} else {
			printf("ReLU is %f, z is %f\n", relu(z), z);
			currentActivation.data[i] = relu(z);
		}

		FreeVector(&weightedActivations);
	}

	if (currentLayer == network.neuronLayerCount - 1) {
		softmax(currentActivation);
		printf("To prove softmax works, this number should be 1: %f\n", VectorSum(currentActivation));
	}

	// The caller has ownership over the initial input so we don't want to free that
	if (currentLayer != 0)
		FreeVector(&input);

	return FeedForward(network, currentActivation, currentLayer + 1);
}

void FreeNetwork(Network* network) {
	for (int i = 0; i < network->neuronLayerCount; i++) {
		FreeVector(&network->biases[i]);
		FreeMatrix(&network->layers[i]);
	}

	free(network->layers);
	free(network->biases);

	network->neuronLayerCount = 0;
}
