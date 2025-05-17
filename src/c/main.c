#include <stdio.h>

#include "network.h"

int main() {
	int layers[] = {5, 4, 3};
	Network network = CreateNetwork(layers, 3);

	for (int i = 0; i < 2; i++) {
		PrintMatrix(network.layers[i]);
		printf("\n");
	}
}
