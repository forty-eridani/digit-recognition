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
	Vector* gradientVectors;
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

	for (int i = 0; i < gradient->count; i++) {
		FreeMatrix(&gradient->gradientMatrices[i]);
		FreeVector(&gradient->gradientVectors[i]);
	}

	gradient->count = 0;

	free(gradient->gradientVectors);
	free(gradient->gradientMatrices);
}

static Gradient averageGradients(Gradient* gradients, int count) {
	assert(gradients);
	assert(count > 0);

	int layerCount = gradients[0].count;

	double scalar = 1 / (double)count;

	Gradient sum = {.count = layerCount, .gradientMatrices = malloc(layerCount * sizeof(Matrix)), 
		.gradientVectors = malloc(layerCount * sizeof(Vector))};

	for (int i = 0; i < sum.count; i++) {
		sum.gradientMatrices[i] = CreateMatrixWithZeros(gradients[0].gradientMatrices[i].rows, gradients[0].gradientMatrices[i].cols);
		sum.gradientVectors[i] = CreateVectorWithZeros(gradients[0].gradientVectors[i].size);
	}

	for (int i = 0; i < count; i++) {
		for (int j = 0; j < gradients[0].count; j++) {
			AddMatricesInPlace(gradients[i].gradientMatrices[j], &sum.gradientMatrices[j]);
			AddVectorsInPlace(gradients[i].gradientVectors[j], &sum.gradientVectors[j]);
		}
	}

	for (int i = 0; i < sum.count; i++) {
		ApplyScalarToMatrixInPlace(&sum.gradientMatrices[i], scalar);
		ApplyScalarToVectorInPlace(&sum.gradientVectors[i], scalar);
	}

	return sum;
}

static void fillArrayRandom(double arr[], int count) {
	for (int i = 0; i < count; i++) {
		arr[i] = ((double)rand() / RAND_MAX - 0.5) * 2;
		// arr[i] = 0.5;
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

static double totalCost(Vector expected, Vector observed) {
	assert(expected.data);
	assert(observed.data);
	assert(observed.size == expected.size);

	double sum = 0.0;

	for (int i = 0; i < observed.size; i++)
		sum += mse(expected.data[i], observed.data[i]);

	return sum;
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

Network CreateNetwork(int* layers, int layerCount, double learningRate) {
	// The network is represented by a bunch of adjacency matrices 
	// Each column represents the layer ahead, and each row represents
	// the neuron that it leads to. Each cell in the matrix represents a
	// weight.
	
	assert(layers);

	srand(time(NULL));
	
	Network network = {.neuronLayerCount = layerCount - 1, 
					   .layers = malloc((layerCount - 1) * sizeof(Matrix)),
	                   .biases = malloc((layerCount - 1) * sizeof(Vector)),
					   .learningRate = learningRate};

	
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

	// printf("Input on layer %d\n", currentLayer);
	// PrintVector(input);

	// We have traversed through all of the layers
	if (currentLayer == network.neuronLayerCount)
		return input;

	if (input.size != network.layers[currentLayer].cols) {
		printf("Input dimensions do not match that of neural network.\n");
		return EMPTY_VECTOR;	
	}

	Vector currentActivation = MultiplyMatrixToVector(network.layers[currentLayer], input);
	AddVectorsInPlace(network.biases[currentLayer], &currentActivation);
	ApplyFunctionToVector(currentActivation, &sigmoid);

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
		Vector* recordedActivations, Matrix* gradientWeights, Vector *gradientBiases, Vector curInfluence, int depth) {
	assert(network.layers);
	assert(network.biases);
	assert(trainingDatum.data);
	assert(expectedOutput.data);
	
	assert(trainingDatum.size == network.inputSize);
	assert(expectedOutput.size == network.outputSize);

	if (gradientWeights == NULL || gradientBiases == NULL) {
		gradientWeights = malloc(network.neuronLayerCount * sizeof(Matrix));
		gradientBiases = malloc(network.neuronLayerCount * sizeof(Vector));

		for (int i = 0; i < network.neuronLayerCount; i++) {
			gradientWeights[i] = CreateMatrixWithZeros(network.layers[i].rows, network.layers[i].cols);
			gradientBiases[i] = CreateVectorWithZeros(network.layers[i].rows);
		}
	}

	if (recordedActivations == NULL) 
		recordedActivations = recordFeedForward(network, NULL, trainingDatum, 0);

	if (curInfluence.data == NULL)
		curInfluence = vectorMsePrime(expectedOutput, recordedActivations[network.neuronLayerCount]);

	if (depth == network.neuronLayerCount) {
		freeVectorArray(recordedActivations, network.neuronLayerCount + 1);
		FreeVector(&curInfluence);
		return (Gradient){.count = depth, .gradientMatrices = gradientWeights, .gradientVectors = gradientBiases};
	}

	int activationIndex = network.neuronLayerCount - depth - 1;
	int inputCount = network.layers[activationIndex].cols;

	Vector nextInfluences = CreateVectorWithZeros(inputCount);

	for (int i = 0; i < network.layers[activationIndex].rows; i++) {
		double* rowAddr = network.layers[activationIndex].data + (i * inputCount);
		Vector weightedActivations = MultiplyVectors(CreateVectorWithElements(rowAddr, inputCount), recordedActivations[activationIndex]);
		double z = VectorSum(weightedActivations) + network.biases[activationIndex].data[i];
		double curNeuronInfluence = curInfluence.data[i];

		FreeVector(&weightedActivations);

		double zPrime = sigmoidPrime(z);
		
		gradientBiases[activationIndex].data[i] = zPrime * curNeuronInfluence;

		for (int j = 0; j < inputCount; j++) {
			// Prev activation never applies to the output layer so this is valid
			double prevActivation = recordedActivations[activationIndex].data[j];

			double weightInfluence = prevActivation * zPrime * curNeuronInfluence;

			gradientWeights[activationIndex].data[i * inputCount + j] += weightInfluence;

			// A little bit confusing naming, but "next" refers to the next neuron to have its
			// gradient computed, not "next" as in the next neuron to process this one's activation
			// (like in feedforward). Also just as a reminder the layers matrix stores weights
			nextInfluences.data[j] += network.layers[activationIndex].data[i * inputCount + j] * zPrime * curNeuronInfluence;
		}
	}

	FreeVector(&curInfluence);

	return computeGradient(network, trainingDatum, expectedOutput, recordedActivations, gradientWeights, gradientBiases, nextInfluences, depth + 1);
}

void BackPropagate(Network network, Vector* trainingInputs, Vector* expectedOutputs, int count) {
	Gradient gradientArr[count];

	for (int i = 0; i < count; i++)
		gradientArr[i] = computeGradient(network, trainingInputs[i], expectedOutputs[i], NULL, NULL, NULL, EMPTY_VECTOR, 0);

	Gradient avg = averageGradients(gradientArr, count);

	assert(avg.count == network.neuronLayerCount);

	for (int i = 0; i < count; i++)
		freeGradient(&gradientArr[i]);

	for (int i = 0; i < avg.count; i++) {
		ApplyScalarToMatrixInPlace(&avg.gradientMatrices[i], -network.learningRate);
		ApplyScalarToVectorInPlace(&avg.gradientVectors[i], -network.learningRate);

		AddMatricesInPlace(avg.gradientMatrices[i], &network.layers[i]);
		AddVectorsInPlace(avg.gradientVectors[i], &network.biases[i]);
	}

	freeGradient(&avg);
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
