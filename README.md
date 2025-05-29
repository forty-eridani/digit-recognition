# digit-recognition

A simple program to recognize handwritten digits from the MNIST database.

## Add Explanation Here Eventually

One day I really hope I add an explanation for how this works here some day...

## Binary File Format For Network

Here is the binary format for the neural network. Sometimes you don't want to leave the perfectly trained weights and models only in memory and have them dissapear after the program as executed.

| 2 bytes (uint16_t) | Layer amount $\times$ 2 bytes (uint16_t) | Neuron amount $\times$ 8 bytes (double) | Layers Matrix Size (excluding input) $\times$ Layer Count $\times$ 8 bytes (double) | 8 bytes (double) |
| - | - | - | - | - |
| Layer amount (including input layer) | Neurons per layer | Biases | Weights | Learning Rate |
