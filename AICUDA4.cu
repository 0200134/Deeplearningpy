#include <stdio.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

// Define constants
#define INPUT_SIZE 784
#define HIDDEN_SIZE 128
#define OUTPUT_SIZE 10
#define LEARNING_RATE 0.01

__global__ void sigmoid(float* x, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        x[idx] = 1.0 / (1.0 + exp(-x[idx]));
    }
}

__global__ void sigmoid_derivative(float* x, float* dx, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dx[idx] = x[idx] * (1.0 - x[idx]);
    }
}

__global__ void forward(float* input, float* weights1, float* bias1, float* weights2, float* bias2, float* output) {
    // Hidden layer
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < HIDDEN_SIZE) {
        float sum = bias1[idx];
        for (int i = 0; i < INPUT_SIZE; ++i) {
            sum += input[i] * weights1[i * HIDDEN_SIZE + idx];
        }
        output[idx] = sum;
    }

    __syncthreads();
    if (idx < HIDDEN_SIZE) {
        sigmoid<<<(HIDDEN_SIZE + 255) / 256, 256>>>(output, HIDDEN_SIZE);
    }

    __syncthreads();
    if (idx < OUTPUT_SIZE) {
        float sum = bias2[idx];
        for (int i = 0; i < HIDDEN_SIZE; ++i) {
            sum += output[i] * weights2[i * OUTPUT_SIZE + idx];
        }
        output[INPUT_SIZE + idx] = sum;
    }

    __syncthreads();
    if (idx < OUTPUT_SIZE) {
        sigmoid<<<(OUTPUT_SIZE + 255) / 256, 256>>>(output + INPUT_SIZE, OUTPUT_SIZE);
    }
}

__global__ void backward(float* input, float* weights1, float* bias1, float* weights2, float* bias2, float* output, float* target) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Error at output layer
    if (idx < OUTPUT_SIZE) {
        output[INPUT_SIZE + idx] = output[INPUT_SIZE + idx] - target[idx];
    }

    __syncthreads();
    if (idx < OUTPUT_SIZE) {
        float grad = output[INPUT_SIZE + idx];
        for (int i = 0; i < HIDDEN_SIZE; ++i) {
            weights2[i * OUTPUT_SIZE + idx] -= LEARNING_RATE * grad * output[i];
        }
        bias2[idx] -= LEARNING_RATE * grad;
    }

    __syncthreads();
    if (idx < HIDDEN_SIZE) {
        float grad = 0.0;
        for (int i = 0; i < OUTPUT_SIZE; ++i) {
            grad += (output[INPUT_SIZE + i] * weights2[idx * OUTPUT_SIZE + i]);
        }
        float output_grad;
        sigmoid_derivative<<<(HIDDEN_SIZE + 255) / 256, 256>>>(output, &output_grad, HIDDEN_SIZE);
        grad *= output_grad;
        for (int i = 0; i < INPUT_SIZE; ++i) {
            weights1[i * HIDDEN_SIZE + idx] -= LEARNING_RATE * grad * input[i];
        }
        bias1[idx] -= LEARNING_RATE * grad;
    }
}

void train(float* input, float* target, float* weights1, float* bias1, float* weights2, float* bias2, int epochs) {
    for (int epoch = 0; epoch < epochs; ++epoch) {
        float* d_input, * d_weights1, * d_bias1, * d_weights2, * d_bias2, * d_output, * d_target;
        cudaMalloc(&d_input, INPUT_SIZE * sizeof(float));
        cudaMalloc(&d_weights1, INPUT_SIZE * HIDDEN_SIZE * sizeof(float));
        cudaMalloc(&d_bias1, HIDDEN_SIZE * sizeof(float));
        cudaMalloc(&d_weights2, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float));
        cudaMalloc(&d_bias2, OUTPUT_SIZE * sizeof(float));
        cudaMalloc(&d_output, (INPUT_SIZE + OUTPUT_SIZE) * sizeof(float));
        cudaMalloc(&d_target, OUTPUT_SIZE * sizeof(float));

        cudaMemcpy(d_input, input, INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_weights1, weights1, INPUT_SIZE * HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_bias1, bias1, HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_weights2, weights2, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_bias2, bias2, OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_target, target, OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);

        forward<<<(HIDDEN_SIZE + 255) / 256, 256>>>(d_input, d_weights1, d_bias1, d_weights2, d_bias2, d_output);
        backward<<<(HIDDEN_SIZE + 255) / 256, 256>>>(d_input, d_weights1, d_bias1, d_weights2, d_bias2, d_output, d_target);

        cudaMemcpy(weights1, d_weights1, INPUT_SIZE * HIDDEN_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(bias1, d_bias1, HIDDEN_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(weights2, d_weights2, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(bias2, d_bias2, OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

        cudaFree(d_input);
        cudaFree(d_weights1);
        cudaFree(d_bias1);
        cudaFree(d_weights2);
        cudaFree(d_bias2);
        cudaFree(d_output);
        cudaFree(d_target);
    }
}

int main() {
    float input[INPUT_SIZE] = { /* Initialize your input data */ };
    float target[OUTPUT_SIZE] = { /* Initialize your target data */ };

    float weights1[INPUT_SIZE * HIDDEN_SIZE];
    float bias1[HIDDEN_SIZE];
    float weights2[HIDDEN_SIZE * OUTPUT_SIZE];
    float bias2[OUTPUT_SIZE];

    // Initialize weights and biases with random values
    for (int i = 0; i < INPUT_SIZE * HIDDEN_SIZE; ++i) {
        weights1[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    for (int i = 0; i < HIDDEN_SIZE; ++i) {
        bias1[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    for (int i = 0; i < HIDDEN_SIZE * OUTPUT_SIZE; ++i) {
        weights2[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    for (int i = 0; i < OUTPUT_SIZE; ++i) {
        bias2[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    int epochs = 1000;
    train(input, target, weights1, bias1, weights2, bias2, epochs);

    // After training, you can use the model for predictions
    // Example of forward pass for predictions:
    float* d_input, * d_weights1, * d_bias1, * d_weights2, * d_bias2, * d_output;
    cudaMalloc(&d_input, INPUT_SIZE * sizeof(float));
    cudaMalloc(&d_weights1, INPUT_SIZE * HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&d_bias1, HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&d_weights2, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float));
    cudaMalloc(&d_bias2, OUTPUT_SIZE * sizeof(float));
    cudaMalloc(&d_output, (INPUT_SIZE + OUTPUT_SIZE) * sizeof(float));

    cudaMemcpy(d_input, input, INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights1, weights1, INPUT_SIZE * HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias1, bias1, HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights2, weights2, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias2, bias2, OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    forward<<<(HIDDEN_SIZE + 255) / 256, 256>>>(d_input, d_weights1, d_bias1, d_weights2, d_bias2, d_output);

    float output[OUTPUT_SIZE];
    cudaMemcpy(output, d_output + INPUT_SIZE, OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < OUTPUT_SIZE; ++i) {
        printf("Output %d: %f\n", i, output[i]);
    }

    cudaFree(d_input);
    cudaFree(d_weights1);
    cudaFree(d_bias1);
    cudaFree(d_weights2);
    cudaFree(d_bias2);
    cudaFree(d_output);

    return 0;
}
