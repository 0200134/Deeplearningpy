#include <iostream>
#include <cmath>

// Neural network parameters
const int inputSize = 3;
const int hiddenSize = 3;
const int outputSize = 1;
const float learningRate = 0.01;

// Device functions for neural network operations
__device__ float sigmoid(float x) {
    return 1.0 / (1.0 + expf(-x));
}

__device__ float sigmoidDerivative(float x) {
    return x * (1.0 - x);
}

// Kernel for forward propagation
__global__ void forwardPropagation(float *inputs, float *weightsIH, float *weightsHO, float *biasH, float *biasO, float *hiddenLayer, float *outputLayer) {
    int tid = threadIdx.x;
    
    if (tid < hiddenSize) {
        float sum = 0.0;
        for (int i = 0; i < inputSize; ++i) {
            sum += inputs[i] * weightsIH[tid * inputSize + i];
        }
        sum += biasH[tid];
        hiddenLayer[tid] = sigmoid(sum);
    }
    
    __syncthreads();
    
    if (tid < outputSize) {
        float sum = 0.0;
        for (int i = 0; i < hiddenSize; ++i) {
            sum += hiddenLayer[i] * weightsHO[tid * hiddenSize + i];
        }
        sum += biasO[tid];
        outputLayer[tid] = sigmoid(sum);
    }
}

// Kernel for backward propagation
__global__ void backwardPropagation(float *inputs, float *weightsIH, float *weightsHO, float *biasH, float *biasO, float *hiddenLayer, float *outputLayer, float *targets) {
    int tid = threadIdx.x;
    
    // Calculate output layer errors
    float outputErrors[outputSize];
    if (tid < outputSize) {
        outputErrors[tid] = targets[tid] - outputLayer[tid];
    }
    
    __syncthreads();
    
    // Update hidden to output weights and biases
    if (tid < hiddenSize) {
        for (int i = 0; i < outputSize; ++i) {
            weightsHO[i * hiddenSize + tid] += learningRate * outputErrors[i] * sigmoidDerivative(outputLayer[i]) * hiddenLayer[tid];
        }
    }
    if (tid < outputSize) {
        biasO[tid] += learningRate * outputErrors[tid] * sigmoidDerivative(outputLayer[tid]);
    }
    
    __syncthreads();
    
    // Calculate hidden layer errors
    float hiddenErrors[hiddenSize];
    if (tid < hiddenSize) {
        hiddenErrors[tid] = 0.0;
        for (int i = 0; i < outputSize; ++i) {
            hiddenErrors[tid] += outputErrors[i] * weightsHO[i * hiddenSize + tid];
        }
    }
    
    __syncthreads();
    
    // Update input to hidden weights and biases
    if (tid < inputSize) {
        for (int i = 0; i < hiddenSize; ++i) {
            weightsIH[i * inputSize + tid] += learningRate * hiddenErrors[i] * sigmoidDerivative(hiddenLayer[i]) * inputs[tid];
        }
    }
    if (tid < hiddenSize) {
        biasH[tid] += learningRate * hiddenErrors[tid] * sigmoidDerivative(hiddenLayer[tid]);
    }
}

int main() {
    float inputs[inputSize] = {1.0, 0.5, -1.0};
    float targets[outputSize] = {0.0};
    
    float weightsIH[hiddenSize * inputSize];
    float weightsHO[outputSize * hiddenSize];
    float biasH[hiddenSize] = {0.0};
    float biasO[outputSize] = {0.0};
    float hiddenLayer[hiddenSize];
    float outputLayer[outputSize];

    // Initialize weights with random values
    for (int i = 0; i < hiddenSize * inputSize; ++i) {
        weightsIH[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    }
    for (int i = 0; i < outputSize * hiddenSize; ++i) {
        weightsHO[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    }

    // Allocate memory on device
    float *d_inputs, *d_weightsIH, *d_weightsHO, *d_biasH, *d_biasO, *d_hiddenLayer, *d_outputLayer, *d_targets;
    cudaMalloc((void**)&d_inputs, inputSize * sizeof(float));
    cudaMalloc((void**)&d_weightsIH, hiddenSize * inputSize * sizeof(float));
    cudaMalloc((void**)&d_weightsHO, outputSize * hiddenSize * sizeof(float));
    cudaMalloc((void**)&d_biasH, hiddenSize * sizeof(float));
    cudaMalloc((void**)&d_biasO, outputSize * sizeof(float));
    cudaMalloc((void**)&d_hiddenLayer, hiddenSize * sizeof(float));
    cudaMalloc((void**)&d_outputLayer, outputSize * sizeof(float));
    cudaMalloc((void**)&d_targets, outputSize * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_inputs, inputs, inputSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weightsIH, weightsIH, hiddenSize * inputSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weightsHO, weightsHO, outputSize * hiddenSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_biasH, biasH, hiddenSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_biasO, biasO, outputSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_targets, targets, outputSize * sizeof(float), cudaMemcpyHostToDevice);

    // Train the network
    for (int epoch = 0; epoch < 1000; ++epoch) {
        forwardPropagation<<<1, max(hiddenSize, outputSize)>>>(d_inputs, d_weightsIH, d_weightsHO, d_biasH, d_biasO, d_hiddenLayer, d_outputLayer);
        backwardPropagation<<<1, max(hiddenSize, outputSize)>>>(d_inputs, d_weightsIH, d_weightsHO, d_biasH, d_biasO, d_hiddenLayer, d_outputLayer, d_targets);
    }

    // Copy results back to host
    cudaMemcpy(outputLayer, d_outputLayer, outputSize * sizeof(float), cudaMemcpyDeviceToHost);

    // Print results
    std::cout << "Output: " << outputLayer[0] << std::endl;

    // Free device memory
    cudaFree(d_inputs);
    cudaFree(d_weightsIH);
    cudaFree(d_weightsHO);
    cudaFree(d_biasH);
    cudaFree(d_biasO);
    cudaFree(d_hiddenLayer);
    cudaFree(d_outputLayer);
    cudaFree(d_targets);

    return 0;
}
