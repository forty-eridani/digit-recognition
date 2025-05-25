#include "network.h"
#include "matrix.h"
#include "vector.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define EMPTY_VECTOR (Vector){.size = 0, .data = NULL}

typedef struct {
	Matrix* gradientMatrices;
	int count;
} Gradient;

static void printGradient(Gradient gradient) {
	printf("Gradient:\n");
	for (int i = 0; i < gradient.count; i++) {
		printf("Layer %d\n", i);
		PrintMatrix(gradient.gradientMatrices[i]);
	}
}

static void freeGradient(Gradient* gradient) {
	assert(gradient);
	assert(gradient->gradientMatrices);

	for (int i = 0; i < gradient->count; i++)
		FreeMatrix(&gradient->gradientMatrices[i]);
}

static void fillArrayRandom(double arr[], int count) {
	srand(time(NULL));

	for (int i = 0; i < count; i++) {
		arr[i] = (double)rand() / RAND_MAX;
	}
}

static double sigmoid(double x) {
	return 1 / (1 + exp(-x));
}

static double sigmoidPrime(double x) {
	return sigmoid(x) * (1 - sigmoid(x));
}

static double mse(double expected, double observed) {
	return (expected - observed) * (expected - observed);
}

static double msePrime(double expected, double observed) {
	// The simplified derivative; if just using chain rule you get 2 * (expected - observed) * -1
	return 2 * (observed - expected);
}

static Vector vectorMsePrime(Vector expected, Vector observed) {
	assert(expected.size == observed.size);
	assert(expected.data);
	assert(observed.data);

	double arr[expected.size];

	for (int i = 0; i < expected.size; i++) {
		arr[i] = msePrime(expected.data[i], observed.data[i]);
	}

	return CreateVector(arr, expected.size);
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

	network.inputSize = layers[0];
	network.outputSize = layers[layerCount - 1];

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
		return EMPTY_VECTOR;	
	}

	Vector currentActivation = MultiplyMatrixToVector(network.layers[currentLayer], input);
	AddVectorsInPlace(network.biases[currentLayer], &currentActivation);

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

	// We still want the caller to have ownership over the original input
	if (depth == 0) {
		activations[depth] = input;
		activations[depth].data = malloc(input.size * sizeof(double));
		memcpy(activations[depth].data, input.data, input.size * sizeof(double));
	} else {
		activations[depth] = input;
	}

	if (depth == network.neuronLayerCount)
		return activations;

	assert(input.size == network.layers[depth].cols);

	if (input.size != network.layers[depth].cols) {
		printf("Input dimensions do not match that of neural network.\n");
		return NULL;	
	}

	Vector currentActivations = MultiplyMatrixToVector(network.layers[depth], input);
	AddVectorsInPlace(network.biases[depth], &currentActivations);

	ApplyFunctionToVector(currentActivations, &sigmoid);

	return recordFeedForward(network, activations, currentActivations, depth + 1);
}

// Pass in `EMPTY_VECTOR` for `curInfluence` initial call so it can be calculated
static Gradient computeGradient(Network network, Vector trainingDatum, Vector expectedOutput, 
		Vector* recordedActivations, Matrix* gradient, Vector curInfluence, int depth) {
	assert(network.layers);
	assert(network.biases);
	assert(trainingDatum.data);
	assert(expectedOutput.data);
	
	assert(trainingDatum.size == network.inputSize);
	assert(expectedOutput.size == network.outputSize);

	if (gradient == NULL) {
		gradient = malloc(network.neuronLayerCount * sizeof(Matrix));

		for (int i = 0; i < network.neuronLayerCount; i++)
			gradient[i] = CreateMatrixWithZeros(network.layers[i].rows, network.layers[i].cols);
	}

	if (recordedActivations == NULL) 
		recordedActivations = recordFeedForward(network, NULL, trainingDatum, 0);

	if (curInfluence.data == NULL)
		curInfluence = vectorMsePrime(expectedOutput, recordedActivations[network.neuronLayerCount]);

	if (depth == network.neuronLayerCount) {
		freeVectorArray(recordedActivations, network.neuronLayerCount + 1);
		return (Gradient){.count = depth, .gradientMatrices = gradient};
	}

	int activationIndex = network.neuronLayerCount - depth - 1;
	int inputCount = network.layers[activationIndex].cols;

	Vector nextInfluences = CreateVectorWithZeros(inputCount);

	for (int i = 0; i < network.layers[activationIndex].rows; i++) {
		double* rowArr = network.layers[activationIndex].data + (i * inputCount);
		double z = VectorSum(CreateVectorWithElements(rowArr, inputCount)) + network.biases[activationIndex].data[i];

		double curNeuronInfluence = curInfluence.data[i];

		for (int j = 0; j < inputCount; j++) {
			// Prev activation never applies to the output layer so this is valid
			double prevActivation = recordedActivations[activationIndex].data[j];

			double weightInfluence = prevActivation * sigmoidPrime(z) * curNeuronInfluence;

			gradient[activationIndex].data[i * inputCount + j] += weightInfluence;

			// A little bit confusing naming, but "next" refers to the next neuron to have its
			// gradient computed, not "next" as in the next neuron to process this one's activation
			// (like in feedforward). Also just as a reminder the layers matrix stores weights
			nextInfluences.data[j] += network.layers[activationIndex].data[i * inputCount + j] * sigmoidPrime(z) * curNeuronInfluence;
		}
	}

	FreeVector(&curInfluence);

	return computeGradient(network, trainingDatum, expectedOutput, recordedActivations, gradient, nextInfluences, depth + 1);
}

void BackPropagate(Network network, Vector* trainingInputs, Vector* expectedOutputs, int count) {
	Vector trainInput = CreateEmptyVector(network.inputSize);
	fillArrayRandom(trainInput.data, trainInput.size);

	Vector expectedOutput = CreateVectorWithZeros(network.outputSize);

	Gradient calculatedGradient = computeGradient(network, trainInput, expectedOutput, NULL, NULL, EMPTY_VECTOR, 0);

	printf("Count: %d\n", calculatedGradient.count);

	printGradient(calculatedGradient);

	Vector* ff = recordFeedForward(network, NULL, trainInput, 0);

	for (int i = 0; i < network.neuronLayerCount + 1; i++) {
		printf("\n");
		PrintVector(ff[i]);
	}

	FreeVector(&trainInput);
	FreeVector(&expectedOutput);
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
