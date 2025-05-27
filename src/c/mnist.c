#include "mnist.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdint.h>

#include "network.h"

#define EPOCH_AMOUNT 128
#define MINI_BATCH_SIZE 100

#define LEARNING_RATE 5.0

static uint16_t swapLsb(uint16_t n) {
	uint8_t most = n;
	uint8_t least = n >> 8;

	return (most << 8) + least;
}

static int readLsbInt(int n) {
	int16_t first = n;
	int16_t next = n >> 16;

	return swapLsb(next) + swapLsb(first);
}

// `size` is a pointer to an int that will be the size of the vector
Vector* readVectorFile(const char* filename, int* size, int magicNumber) {
	FILE* fptr = fopen(filename, "rb");

	if (!fptr) {
		printf("Could not open file '%s'.\n", filename);
		return NULL;
	}
	
	int metadata[4];
	size_t bytesRead = fread(metadata, sizeof(int), 4, fptr);

	for (int i = 0; i < 4; i++)
		metadata[i] = readLsbInt(metadata[i]);

	if (metadata[0] != magicNumber) {
		printf("Magic number '%d' does not match read magic number of '%d'.\n", magicNumber, metadata[0]);
		return NULL;
	}

	int imageCount = metadata[1];
	int imageHeight = metadata[2];
	int imageWidth = metadata[3];
	
	Vector* arr = malloc(imageCount * sizeof(Vector));

	*size = imageCount;

	for (int i = 0; i < imageCount; i++) {
		uint8_t* buf = malloc(imageHeight * imageWidth * sizeof(uint8_t));

		fread(buf, sizeof(uint8_t), imageHeight * imageWidth, fptr);

		Vector v = CreateEmptyVector(imageHeight * imageWidth);

		for (int j = 0; j < imageHeight * imageWidth; j++)
			v.data[j] = (double)buf[j] / 255;
		
		arr[i] = v;
		free(buf);
	}

	fclose(fptr);

	return arr;
}

int8_t* readInputArray(const char* filename, int* size, int magicNumber) {
	FILE* fptr = fopen(filename, "rb");

	if (!fptr) {
		printf("Could not open file '%s'.\n", filename);
		return NULL;
	}

	int metadata[2];

	fread(metadata, sizeof(int), 2, fptr);

	for (int i = 0; i < 2; i++)
		metadata[i] = readLsbInt(metadata[i]);

	int readMagicNumber = metadata[0];
	int labelCount = metadata[1];

	if (magicNumber != readMagicNumber) {
		printf("Magic number '%d' does not match read magic number of '%d'.\n", magicNumber, metadata[0]);
		return NULL;
	}

	int8_t* arr = malloc(labelCount * sizeof(int8_t));

	*size = fread(arr, sizeof(int8_t), labelCount, fptr);

	fclose(fptr);

	return arr;
}

void shuffleVectorArray(Vector* array, int size) {
	for (int i = size - 1; i > 0; i--) {
		int j = rand() % i;

		Vector tmp = array[i];
		array[i] = array[j];
		array[j] = tmp;
	}
}

// Generates a 10 element vector where `n` represents the index of the element that is `1.0`
Vector generateVectorFromNumber(int8_t n) {
	assert(n >= 0 && n < 10);

	Vector v = CreateVectorWithZeros(10);
	v.data[n] = 1.0;

	return v;
}

Vector vectorCost(Vector expected, Vector observed) {
	assert(expected.data);
	assert(observed.data);
	assert(observed.size == expected.size);

	Vector cost = ApplyScalarToVector(observed, -1.0);
	AddVectorsInPlace(expected, &cost);
	MultiplyVectorsInPlace(cost, &cost);

	return cost;
}

int8_t getObservedDigit(Vector v) {
	double max = v.data[0];
	int8_t n = 0;

	for (int i = 1; i < v.size; i++) {
		if (v.data[i] > max) {
			max = v.data[i];
			n = i;
		}
	}

	return n;
}

void RecognizeDigits() {
	int trainingSize;
	int labelSize;
	
	// Magic number is just info about how the data is stored
	Vector* trainingInputs = readVectorFile("../input/train-images.idx3-ubyte", &trainingSize, 0x0803);
	int8_t* labels = readInputArray("../input/train-labels.idx1-ubyte", &labelSize, 0x0801);

	assert(trainingSize == labelSize);

	Vector* vectorLabels = malloc(labelSize * sizeof(Vector));

	for (int i = 0; i < labelSize; i++) {
		vectorLabels[i] = generateVectorFromNumber(labels[i]);
	}

	// for (int i = 0; i < 28 * 28; i++)
	// 	if (trainingInputs[0].data[i] > 0)
	// 		printf("%f\n", trainingInputs[0].data[i]);

	srand(time(NULL));

	shuffleVectorArray(trainingInputs, trainingSize);
	shuffleVectorArray(vectorLabels, labelSize);

	int layers[] = {28 * 28, 16, 16, 10};

	Network network = CreateNetwork(layers, 4, LEARNING_RATE);

	for (int i = 0; i < EPOCH_AMOUNT; i++) {
		
		for (int i = 0; i < trainingSize / MINI_BATCH_SIZE; i++)
			BackPropagate(network, trainingInputs + (i * MINI_BATCH_SIZE), vectorLabels + (i * MINI_BATCH_SIZE), MINI_BATCH_SIZE);

		printf("Epoch %d completed.\n", i);
	}

	Vector sum = CreateVectorWithZeros(10);
	int correctlyIdentified = 0;

	for (int i = 0; i < trainingSize; i++) {
		Vector feedForward = FeedForward(network, trainingInputs[i], 0);
		Vector cost = vectorCost(vectorLabels[i], feedForward);

		AddVectorsInPlace(cost, &sum);

		if (getObservedDigit(feedForward) == getObservedDigit(vectorLabels[i]))
				correctlyIdentified++;

		FreeVector(&feedForward);
		FreeVector(&cost);
	}

	ApplyScalarToVectorInPlace(&sum, 1.0 / (double)EPOCH_AMOUNT);

	printf("Cost after training: %f, Percentage of training data correctly identified: %f%%.\n", VectorSum(sum),
			((double)correctlyIdentified / (double)trainingSize) * 100.0);

	FreeVector(&sum);

	for (int i = 0; i < trainingSize; i++) {
		FreeVector(&trainingInputs[i]);
		FreeVector(&vectorLabels[i]);
	}

	free(trainingInputs);
	free(labels);
	free(vectorLabels);

	FreeNetwork(&network);
}
