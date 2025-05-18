#include <stdio.h>

#include "network.h"

int main() {
	int layers[] = {5, 4, 3};
	Network network = CreateNetwork(layers, 3);

	for (int i = 0; i < 2; i++) {
		PrintMatrix(network.layers[i]);
		PrintVector(network.biases[i]);
		printf("\n");
	}

	double vectorMatrixInput[] = {1.0, 0.5, 0.0, 0.23, 0.65};
	Vector input = CreateVectorWithElements(vectorMatrixInput, 5);

	Vector output = FeedForward(network, input, 0);

	PrintVector(output);

	FreeVector(&output);
	FreeNetwork(&network);
}
