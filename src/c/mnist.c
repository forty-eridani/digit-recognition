#include "mnist.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdint.h>

#include "network.h"

#define EPOCH_AMOUNT 50
#define MINI_BATCH_SIZE 50

#define LEARNING_RATE 0.7

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

void shuffleIntArray(int* array, int size) {
	for (int i = size - 1; i > 0; i--) {
		int j = rand() % i;

		int tmp = array[i];
		array[i] = array[j];
		array[j] = tmp;
	}
}

// Generates a 10 element vector where `n` represents the index of the element that is `1.0`
Vector generateVectorFromNumber(int8_t n) {
	if (n < 0 || n >= 10)
		printf("%d\n", n);
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
	int8_t* trainingLabels = readInputArray("../input/train-labels.idx1-ubyte", &labelSize, 0x0801);

	assert(trainingSize == labelSize);

	Vector* trainingVectorLabels = malloc(labelSize * sizeof(Vector));

	for (int i = 0; i < labelSize; i++) {
		trainingVectorLabels[i] = generateVectorFromNumber(trainingLabels[i]);
	}

	// for (int i = 0; i < 28 * 28; i++)
	// 	if (trainingInputs[0].data[i] > 0)
	// 		printf("%f\n", trainingInputs[0].data[i]);

	srand(time(NULL));

	int indices[trainingSize];

	int layers[] = {28 * 28, 16, 16, 10};

	Network network = CreateNetwork(layers, 4, LEARNING_RATE);

	for (int i = 0; i < EPOCH_AMOUNT; i++) {

		for (int i = 0; i < trainingSize; i++)
			indices[i] = i;

		shuffleIntArray(indices, trainingSize);
		
		for (int i = 0; i < trainingSize / MINI_BATCH_SIZE; i++) {
			Vector miniBatchImages[MINI_BATCH_SIZE];
			Vector miniBatchLabels[MINI_BATCH_SIZE];

			for (int j = 0; j < MINI_BATCH_SIZE; j++) {
				miniBatchImages[j] = trainingInputs[indices[i * MINI_BATCH_SIZE + j]];
				miniBatchLabels[j] = trainingVectorLabels[indices[i * MINI_BATCH_SIZE + j]];
			}

			BackPropagate(network, miniBatchImages, miniBatchLabels, MINI_BATCH_SIZE);
		}

		printf("Epoch %d completed.\n", i);
	}

	Vector sum = CreateVectorWithZeros(10);
	int correctlyIdentified = 0;

	for (int i = 0; i < trainingSize; i++) {
		Vector feedForward = FeedForward(network, trainingInputs[i], 0);
		Vector cost = vectorCost(trainingVectorLabels[i], feedForward);

		AddVectorsInPlace(cost, &sum);

		if (getObservedDigit(feedForward) == getObservedDigit(trainingVectorLabels[i]))
				correctlyIdentified++;

		FreeVector(&feedForward);
		FreeVector(&cost);
	}

	ApplyScalarToVectorInPlace(&sum, 1.0 / (double)EPOCH_AMOUNT);

	printf("Cost after training: %f, Percentage of training data correctly identified: %f%%.\n", VectorSum(sum),
			((double)correctlyIdentified / (double)trainingSize) * 100.0);

	FreeVector(&sum);

	int testImageSize;
	int testLabelSize;

	Vector* testImages = readVectorFile("../input/t10k-images.idx3-ubyte", &testImageSize, 0x0803);
	int8_t* testLabels = readInputArray("../input/t10k-labels.idx1-ubyte", &testLabelSize, 0x0801);

	assert(testImageSize == testLabelSize);

	printf("%d is size\n", testImageSize);

	Vector* testVectorLabels = malloc(labelSize * sizeof(Vector));

	for (int i = 0; i < testLabelSize ; i++) {
		testVectorLabels[i] = generateVectorFromNumber(testLabels[i]);
	}

	correctlyIdentified = 0;

	for (int i = 0; i < testImageSize; i++) {
		Vector feedForward = FeedForward(network, testImages[i], 0);

		if (getObservedDigit(feedForward) == getObservedDigit(testVectorLabels[i]))
				correctlyIdentified++;

		FreeVector(&feedForward);
	}

	printf("Correctly identified test labels: %f%%.\n", ((double)correctlyIdentified / (double)testImageSize) * 100.0);

	for (int i = 0; i < trainingSize; i++) {
		FreeVector(&trainingInputs[i]);
		FreeVector(&trainingVectorLabels[i]);
	}

	for (int i = 0; i < testImageSize; i++) {
		FreeVector(&testImages[i]);
		FreeVector(&testVectorLabels[i]);
	}

	free(trainingInputs);
	free(trainingVectorLabels);
	free(trainingLabels);
	
	free(testVectorLabels);
	free(testLabels);
	free(testImages);

	FreeNetwork(&network);
}
