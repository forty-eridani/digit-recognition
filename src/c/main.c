#include <stdio.h>

#include "network.h"

int main() {
	int layers[] = {5, 4, 3};
	Network network = CreateNetwork(layers, 3);

	double vectorMatrixInput[5] = {1.0, 2.0, 3.0, 4.0, 5.0};
	Vector input = CreateVectorWithElements(vectorMatrixInput, 5);

	BackPropagate(network, &input, NULL, 1);

	FreeNetwork(&network);
}
