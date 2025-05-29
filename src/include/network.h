#ifndef NETWORK_H_
#define NETWORK_H_

#include "matrix.h"

// Neural network, made of adjacency lists
typedef struct {
	// A matrix and bias per neuron layer
	Matrix* layers;
	Vector* biases;

	// Does not include input neurons
	int neuronLayerCount;

	// Stored in matrix dimensions in layers as well, but code becomes unclear
	int inputSize;
	int outputSize;

	double learningRate;
} Network;

// Calculates the MSE of the observed value (y hat) and the expected value (y) 
Vector CalculateVectorCost(Vector observed, Vector expected);

// Initializes a neural network with random weights and biases. Layer array
// includes input size
Network CreateNetwork(int* layers, int layerCount, double learningRate);

// Returns a vector representing the output. `currentLayer` should start at zero if input is truly input. Caller is still in charge of freeing input
Vector FeedForward(Network network, Vector input, int currentLayer);

// Back propagate with the given array of training inputs and expected outputs
void BackPropagate(Network network, Vector* trainingInputs, Vector* expectedOutputs, int count);

// Save the weights and biases and layers to a file in a custom binary format
void SaveParametersToFile(Network network, const char* filename);

// Loads the weights and biases and contructs the network from them
Network LoadParametersFromFile(const char* filename);

// Frees the memory in the network and sets all 
void FreeNetwork(Network* network);

#endif
