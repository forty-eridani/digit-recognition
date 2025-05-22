#include "network.h"
#include "matrix.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define ERROR_OUTPUT (Vector){.size = 0, .data = NULL}

static void fillArrayRandom(double arr[], int count) {
	srand(time(NULL));

	for (int i = 0; i < count; i++) {
		arr[i] = 1.0;
		// arr[i] = (double)rand() / RAND_MAX;
	}
}

static double relu(double value) {
	return value > 0 ? value : 0;
}

static void vectorRelu(Vector vector) {
	for (int i = 0; i < vector.size; i++)
		vector.data[i] = relu(vector.data[i]);
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
	assert(currentLayer >= 0 && currentLayer <= network.neuronLayerCount);

	printf("Input on layer %d\n", currentLayer);
	PrintVector(input);

	// We have traversed through all of the layers
	if (currentLayer == network.neuronLayerCount)
		return input;

	if (input.size != network.layers[currentLayer].cols) {
		printf("Input dimensions do not match that of neural network.\n");
		return ERROR_OUTPUT;	
	}

	Vector currentActivation = MultiplyMatrixToVector(network.layers[currentLayer], input);
	AddVectorsInPlace(network.biases[currentLayer], &currentActivation);

	if (currentLayer < network.neuronLayerCount - 1) 
		vectorRelu(currentActivation);
	else {
		softmax(currentActivation);
		// printf("To prove softmax works, this number should be 1: %f\n", VectorSum(currentActivation));
	}

	// The caller has ownership over the initial input so we don't want to free that
	if (currentLayer != 0)
		FreeVector(&input);

	return FeedForward(network, currentActivation, currentLayer + 1);
}

static void freeVectorArray(Vector* array, int size) {
	for (int i = 0; i < size; i++) 
		FreeVector(&array[i]);

	free(array);
}

static Vector copyVector(Vector src) {
	Vector new = src;
	new.data = malloc(src.size * sizeof(double));
	memcpy(new.data, src.data, src.size * sizeof(double));

	return new;
}

static Vector* recordFeedForward(Network network, Vector* activations, Vector input, int depth) {
	assert(network.biases);
	assert(network.layers);
	assert(input.data);
	assert(depth >= 0 && depth <= network.neuronLayerCount);

	if (activations == NULL)
		// We must include the input vector
		activations = malloc((network.neuronLayerCount + 1) * sizeof(Vector));

	if (depth == 0) {
		activations[depth] = input;
		activations[depth].data = malloc(input.size * sizeof(double));
		memcpy(activations[depth].data, input.data, input.size * sizeof(double));
	} else {
		activations[depth] = input;
	}

	if (depth == network.neuronLayerCount)
		return activations;

	if (input.size != network.layers[depth].cols) {
		printf("Input dimensions do not match that of neural network.\n");
		return NULL;	
	}

	Vector currentActivations = MultiplyMatrixToVector(network.layers[depth], input);
	AddVectorsInPlace(network.biases[depth], &currentActivations);

	if (depth < network.neuronLayerCount - 1) 
		vectorRelu(currentActivations);
	else
		softmax(currentActivations);

	return recordFeedForward(network, activations, currentActivations, depth + 1);
}


void BackPropagate(Network network, Vector* trainingInputs, Vector* expectedOutputs, int count) {
	Vector* feedForwardData = recordFeedForward(network, NULL, *trainingInputs, 0);

	for (int i = 0; i < network.neuronLayerCount + 1; i++) {
		PrintVector(feedForwardData[i]);
		printf("\n");
	}

	freeVectorArray(feedForwardData, network.neuronLayerCount + 1);
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
