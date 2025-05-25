#include <assert.h>
#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include "network.h"

#define POINT_TRAIN_AMOUNT 1024
#define POINT_TEST_AMOUNT 1024 

#define MAX_DISTANCE 1.0
#define THRESHOLD_DISTANCE 0.5

#define PI 3.141592653589793

#define MINI_BATCH_SIZE 16
#define EPOCH_COUNT 100

double RaiseToTwo(double x) {
	return x * x;
}

void ComputeVectorCost(Vector expected, Vector* observed) {
	assert(expected.data);
	assert(observed);
	assert(observed->data);
	assert(observed->size == expected.size);

	ApplyScalarToVectorInPlace(observed, -1);
	AddVectorsInPlace(expected, observed);
	ApplyFunctionToVector(*observed, &RaiseToTwo);

}

double ComputeTotalCost(Vector expected, Vector observed) {
	assert(expected.data);
	assert(expected.data);
	assert(observed.size == expected.size);

	Vector costVector = ApplyScalarToVector(observed, -1);
	AddVectorsInPlace(expected, &costVector);
	ApplyFunctionToVector(costVector, &RaiseToTwo);

	double sum = VectorSum(costVector);
	FreeVector(&costVector);

	return sum;
}

// A little test program that will utilize the neural network to classify points
void PointClassification() {
	srand(time(NULL));

	Vector trainingPointInputs[POINT_TRAIN_AMOUNT];
	Vector trainingPointOutputs[POINT_TRAIN_AMOUNT];

	Vector testPointInputs[POINT_TRAIN_AMOUNT];
	Vector testPointOuputs[POINT_TRAIN_AMOUNT];

	const double innerClass[] = {0.0, 1.0};
	const double outerClass[] = {1.0, 0.0};

	for (int i = 0; i < POINT_TRAIN_AMOUNT; i++) {
		double distance = ((double)rand() / RAND_MAX) * MAX_DISTANCE;
		double angle = ((double)rand() / RAND_MAX) * 2 * PI;

		const double vectorArray[2] = {
			distance * cos(angle),
			distance * sin(angle)
		};

		trainingPointInputs[i] = CreateVector(vectorArray, 2);

		if (distance > THRESHOLD_DISTANCE) 
			trainingPointOutputs[i] = CreateVector(outerClass, 2);
		else 
			trainingPointOutputs[i] = CreateVector(innerClass, 2);
	}

	for (int i = 0; i < POINT_TEST_AMOUNT; i++) {
		double distance = ((double)rand() / RAND_MAX) * MAX_DISTANCE;
		double angle = ((double)rand() / RAND_MAX) * 2 * PI;

		const double vectorArray[2] = {
			distance * cos(angle),
			distance * sin(angle)
		};

		testPointInputs[i] = CreateVector(vectorArray, 2);

		if (distance > THRESHOLD_DISTANCE) 
			testPointOuputs[i] = CreateVector(outerClass, 2);
		else 
			testPointOuputs[i] = CreateVector(innerClass, 2);
	}

	int layers[] = {2, 5, 5, 2};

	Network network = CreateNetwork(layers, 4, 1.0);

	for (int i = 0; i < EPOCH_COUNT; i++) {
		// We use mini-batch descent
		for (int j = 0; j < POINT_TRAIN_AMOUNT / MINI_BATCH_SIZE; j++) {
			BackPropagate(network, trainingPointInputs + (j * MINI_BATCH_SIZE), trainingPointOutputs + (j * MINI_BATCH_SIZE), MINI_BATCH_SIZE);
		}

		Vector cost = CreateVectorWithZeros(2);

		for (int i = 0; i < POINT_TRAIN_AMOUNT; i++) {
			Vector feedForwardResult = FeedForward(network, trainingPointInputs[i], 0);
			ComputeVectorCost(trainingPointOutputs[i], &feedForwardResult);
			AddVectorsInPlace(feedForwardResult, &cost);
			FreeVector(&feedForwardResult);
		}

		ApplyScalarToVectorInPlace(&cost, 1.0 / (double)POINT_TRAIN_AMOUNT);
		
		printf("Cost after epoch %d: %f\n", i, VectorSum(cost));
	
		FreeVector(&cost);
	}

	int correctlyIdentified = 0;

	for (int i = 0; i < POINT_TRAIN_AMOUNT; i++) {
		Vector output = FeedForward(network, trainingPointInputs[i], 0);
		Vector expectedOutput = trainingPointOutputs[i];

		if ((expectedOutput.data[0] > expectedOutput.data[1] && output.data[0] > output.data[1])
			|| (expectedOutput.data[0] < expectedOutput.data[1] && output.data[0] < output.data[1]))
			correctlyIdentified++;

		FreeVector(&output);
	}

	printf("Percentage of training data correctly identified: %f%%\n", ((double)correctlyIdentified / (double)POINT_TRAIN_AMOUNT) * 100.0);

	correctlyIdentified = 0;

	for (int i = 0; i < POINT_TEST_AMOUNT; i++) {
		Vector output = FeedForward(network, testPointInputs[i], 0);
		Vector expectedOutput = testPointOuputs[i];

		if ((expectedOutput.data[0] > expectedOutput.data[1] && output.data[0] > output.data[1])
			|| (expectedOutput.data[0] < expectedOutput.data[1] && output.data[0] < output.data[1]))
			correctlyIdentified++;

		FreeVector(&output);
	}

	printf("Percentage of testing data correctly identified: %f%%\n", ((double)correctlyIdentified / (double)POINT_TRAIN_AMOUNT) * 100.0);

	FreeNetwork(&network);

	for (int i = 0; i < POINT_TRAIN_AMOUNT; i++) {
		FreeVector(&trainingPointInputs[i]);
		FreeVector(&trainingPointOutputs[i]);
	}

	for (int i = 0; i < POINT_TEST_AMOUNT; i++) {
		FreeVector(&testPointInputs[i]);
		FreeVector(&testPointOuputs[i]);
	}
}

int main() {
	// Vector zeros = CreateVectorWithZeros(3);
	//
	// int layers[] = {3, 2, 3};
	//
	// Network network = CreateNetwork(layers, 3, 0.5);
	//
	// Vector v = FeedForward(network, zeros, 0);
	//
	// BackPropagate(network, &zeros, &zeros, 1);
	//
	// FreeNetwork(&network);
	// FreeVector(&zeros);
	// FreeVector(&v);
	
	PointClassification();
}
