import Foundation

// Activation function
func sigmoid(_ x: Double) -> Double {
    return 1 / (1 + exp(-x))
}

// Derivative of the sigmoid function
func sigmoidDerivative(_ x: Double) -> Double {
    return x * (1 - x)
}

// Define the Neural Network class
class NeuralNetwork {
    var inputLayer: [Double]
    var weights: [[Double]]
    var biases: [Double]
    var learningRate: Double

    init(inputSize: Int, hiddenSize: Int, outputSize: Int, learningRate: Double) {
        self.inputLayer = [Double](repeating: 0.0, count: inputSize)
        self.weights = [
            [Double](repeating: 0.0, count: hiddenSize), // Input to hidden layer weights
            [Double](repeating: 0.0, count: outputSize)   // Hidden to output layer weights
        ]
        self.biases = [Double](repeating: 0.0, count: hiddenSize + outputSize)
        self.learningRate = learningRate

        // Randomize weights and biases
        for i in 0..<weights.count {
            for j in 0..<weights[i].count {
                weights[i][j] = Double.random(in: -1.0...1.0)
            }
        }
        for i in 0..<biases.count {
            biases[i] = Double.random(in: -1.0...1.0)
        }
    }

    // Forward propagation
    func forward(_ inputs: [Double]) -> [Double] {
        inputLayer = inputs
        var hiddenLayer = [Double](repeating: 0.0, count: weights[0].count)
        
        // Calculate hidden layer activations
        for i in 0..<hiddenLayer.count {
            for j in 0..<inputLayer.count {
                hiddenLayer[i] += inputLayer[j] * weights[0][i]
            }
            hiddenLayer[i] = sigmoid(hiddenLayer[i] + biases[i])
        }

        var outputLayer = [Double](repeating: 0.0, count: weights[1].count)
        // Calculate output layer activations
        for i in 0..<outputLayer.count {
            for j in 0..<hiddenLayer.count {
                outputLayer[i] += hiddenLayer[j] * weights[1][i]
            }
            outputLayer[i] = sigmoid(outputLayer[i] + biases[hiddenLayer.count + i])
        }
        return outputLayer
    }

    // Backward propagation
    func backward(_ targetOutput: [Double]) {
        var outputError = [Double](repeating: 0.0, count: targetOutput.count)
        var outputDelta = [Double](repeating: 0.0, count: targetOutput.count)
        
        // Calculate output layer error and delta
        let outputLayer = forward(inputLayer)
        for i in 0..<targetOutput.count {
            outputError[i] = targetOutput[i] - outputLayer[i]
            outputDelta[i] = outputError[i] * sigmoidDerivative(outputLayer[i])
        }

        var hiddenError = [Double](repeating: 0.0, count: weights[0].count)
        var hiddenDelta = [Double](repeating: 0.0, count: weights[0].count)
        
        // Calculate hidden layer error and delta
        for i in 0..<hiddenError.count {
            for j in 0..<outputDelta.count {
                hiddenError[i] += outputDelta[j] * weights[1][j]
            }
            hiddenDelta[i] = hiddenError[i] * sigmoidDerivative(inputLayer[i])
        }

        // Update weights and biases
        for i in 0..<weights[0].count {
            for j in 0..<inputLayer.count {
                weights[0][i] += learningRate * hiddenDelta[i] * inputLayer[j]
            }
            biases[i] += learningRate * hiddenDelta[i]
        }
        for i in 0..<weights[1].count {
            for j in 0..<hiddenLayer.count {
                weights[1][i] += learningRate * outputDelta[i] * hiddenLayer[j]
            }
            biases[hiddenLayer.count + i] += learningRate * outputDelta[i]
        }
    }

    // Train the network
    func train(inputs: [[Double]], targetOutputs: [[Double]], iterations: Int) {
        for _ in 0..<iterations {
            for (input, target) in zip(inputs, targetOutputs) {
                forward(input)
                backward(target)
            }
        }
    }
}

// Example usage
let nn = NeuralNetwork(inputSize: 3, hiddenSize: 4, outputSize: 1, learningRate: 0.1)
let inputs = [[0.0, 0.0, 1.0], [1.0, 1.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]]
let targetOutputs = [[0.0], [1.0], [1.0], [0.0]]
nn.train(inputs: inputs, targetOutputs: targetOutputs, iterations: 10000)

let output = nn.forward([1.0, 0.0, 0.0])
print("Output for [1.0, 0.0, 0.0]: \(output)")
