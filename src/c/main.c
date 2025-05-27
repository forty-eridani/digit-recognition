#include <assert.h>
#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include "network.h"
#include "mnist.h"

#define POINT_TRAIN_AMOUNT 1024
#define POINT_TEST_AMOUNT 1024 

#define MAX_DISTANCE 1.0
#define THRESHOLD_DISTANCE 0.5

#define PI 3.141592653589793

#define MINI_BATCH_SIZE 16
#define EPOCH_COUNT 1000

#define WINDOW_WIDTH 512
#define WINDOW_HEIGHT 512

#define SDL_MAIN_USE_CALLBACKS 1  /* use the callbacks instead of main() */
#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>

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

	Network network = CreateNetwork(layers, 4, 0.4);

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

static Vector trainingPointInputs[POINT_TRAIN_AMOUNT];
static Vector trainingPointOutputs[POINT_TRAIN_AMOUNT];

static Vector testPointInputs[POINT_TRAIN_AMOUNT];
static Vector testPointOuputs[POINT_TRAIN_AMOUNT];

void Epoch(Network network, int epochIndex) {
	for (int j = 0; j < POINT_TRAIN_AMOUNT / MINI_BATCH_SIZE; j++) {
		BackPropagate(network, trainingPointInputs + (j * MINI_BATCH_SIZE), trainingPointOutputs + (j * MINI_BATCH_SIZE), MINI_BATCH_SIZE);
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

	printf("Percentage of training data correctly identified on epoch %d: %f%%\n", epochIndex, ((double)correctlyIdentified / (double)POINT_TRAIN_AMOUNT) * 100.0);
}

typedef struct {
	SDL_Window* window;
	SDL_Renderer* renderer;
	Network network;
} AppState;

static int it = 0;

SDL_AppResult SDL_AppInit(void **appstate, int arc, char* argv[]) {
	srand(time(NULL));

	SDL_SetAppMetadata("Decision Boundary Visualization", "1.0", "com.forty-eridani.visual");

	if (!SDL_Init(SDL_INIT_VIDEO)) {
		SDL_Log("Error Initializing SDL: %s", SDL_GetError());
		return SDL_APP_FAILURE;
	}

	AppState* as = malloc(sizeof(AppState));
	*appstate = as;

	int layers[] = {2, 4, 4, 2};

	as->network = CreateNetwork(layers, 4, 0.1);

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

	if (!SDL_CreateWindowAndRenderer("Decision Boundary Visualization", WINDOW_WIDTH, WINDOW_HEIGHT, 0, &as->window, &as->renderer)) {
		SDL_Log("Error initializing window and/or renderer: %s", SDL_GetError());
		return SDL_APP_FAILURE;
	}

	return SDL_APP_CONTINUE;
}

SDL_AppResult SDL_AppEvent(void* appstate, SDL_Event* event) {
	if (event->type == SDL_EVENT_QUIT) {
		return SDL_APP_SUCCESS;
	}

	return SDL_APP_CONTINUE;
}

SDL_AppResult SDL_AppIterate(void* appstate) {
	AppState* as = (AppState*)appstate;

	Epoch(as->network, it++);

	SDL_Texture* texture = SDL_CreateTexture( as->renderer, SDL_PIXELFORMAT_RGBA8888, SDL_TEXTUREACCESS_STREAMING, WINDOW_WIDTH, WINDOW_HEIGHT);

	unsigned char* pixels;
	int pitch;

	SDL_LockTexture( texture, NULL, (void**)&pixels, &pitch );

	printf("pitch %d\n", pitch);

	for (int y = 0; y < WINDOW_HEIGHT; y++) {
		for (int x = 0; x < WINDOW_WIDTH; x++) {
			double xCoord = ((double)x - (double)(WINDOW_WIDTH / 2.0)) / ((double)WINDOW_WIDTH / 2.0);
			double yCoord = ((double)y - (double)(WINDOW_HEIGHT / 2.0)) / ((double)WINDOW_HEIGHT / 2.0);

			double arr[] = {xCoord, yCoord};

			Vector result = FeedForward(as->network, CreateVectorWithElements(arr, 2), 0);

			uint8_t red = result.data[0] * 255;
			uint8_t green = 0xff;
			uint8_t blue = result.data[1] * 255;
			
			uint32_t color = (0xff) + ((uint16_t)(result.data[1] * 255) << 8) + ((uint32_t)(result.data[0] * 255) << 24);

			uint32_t* pixelsAsPixels = (uint32_t*)pixels;
			pixelsAsPixels[y * WINDOW_WIDTH + x] = color;

			FreeVector(&result);
		}
	}

	SDL_UnlockTexture(texture);

	SDL_RenderClear(as->renderer);
	SDL_RenderTexture(as->renderer, texture, NULL, NULL);
	SDL_RenderPresent(as->renderer);

	SDL_DestroyTexture(texture);

	return SDL_APP_CONTINUE;
}

void SDL_AppQuit(void* appstate, SDL_AppResult result) {
	FreeNetwork(&((AppState*)appstate)->network);
	free(appstate);

	for (int i = 0; i < POINT_TRAIN_AMOUNT; i++) {
		FreeVector(&trainingPointInputs[i]);
		FreeVector(&trainingPointOutputs[i]);
	}

	for (int i = 0; i < POINT_TEST_AMOUNT; i++) {
		FreeVector(&testPointInputs[i]);
		FreeVector(&testPointOuputs[i]);
	}
}