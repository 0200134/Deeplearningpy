import Foundation

// Activation function and its derivative
struct ActivationFunction {
    static func sigmoid(_ x: Double) -> Double {
        return 1 / (1 + exp(-x))
    }

    static func sigmoidDerivative(_ x: Double) -> Double {
        return x * (1 - x)
    }
}

// Neural Network class
class NeuralNetwork {
    private var weightsInputHidden: [[Double]]
    private var weightsHiddenOutput: [Double]
    private var biasesHidden: [Double]
    private var biasesOutput: Double
    private var learningRate: Double
    
    init(inputSize: Int, hiddenSize: Int, learningRate: Double) {
        self.learningRate = learningRate
        self.weightsInputHidden = Array(repeating: Array(repeating: 0.0, count: hiddenSize), count: inputSize)
        self.weightsHiddenOutput = Array(repeating: 0.0, count: hiddenSize)
        self.biasesHidden = Array(repeating: 0.0, count: hiddenSize)
        self.biasesOutput = 0.0
        
        // Initialize weights and biases
        for i in 0..<inputSize {
            for j in 0..<hiddenSize {
                weightsInputHidden[i][j] = Double.random(in: -1.0...1.0)
            }
        }
        for i in 0..<hiddenSize {
            weightsHiddenOutput[i] = Double.random(in: -1.0...1.0)
            biasesHidden[i] = Double.random(in: -1.0...1.0)
        }
        biasesOutput = Double.random(in: -1.0...1.0)
    }
    
    // Forward propagation
    func forward(_ inputs: [Double]) -> Double {
        let hiddenOutputs = inputs.enumerated().map { (i, input) in
            weightsInputHidden[i].enumerated().map { (j, weight) in
                input * weight
            }
        }.reduce(Array(repeating: 0.0, count: weightsHiddenOutput.count)) { (sum, next) in
            zip(sum, next).map(+)
        }.enumerated().map { (i, sum) in
            ActivationFunction.sigmoid(sum + biasesHidden[i])
        }
        
        let finalOutput = hiddenOutputs.enumerated().reduce(0.0) { (sum, next) in
            sum + next.element * weightsHiddenOutput[next.offset]
        }
        
        return ActivationFunction.sigmoid(finalOutput + biasesOutput)
    }
    
    // Backward propagation
    func backward(_ inputs: [Double], _ expectedOutput: Double) {
        let hiddenOutputs = inputs.enumerated().map { (i, input) in
            weightsInputHidden[i].enumerated().map { (j, weight) in
                input * weight
            }
        }.reduce(Array(repeating: 0.0, count: weightsHiddenOutput.count)) { (sum, next) in
            zip(sum, next).map(+)
        }.enumerated().map { (i, sum) in
            ActivationFunction.sigmoid(sum + biasesHidden[i])
        }
        
        let finalOutput = hiddenOutputs.enumerated().reduce(0.0) { (sum, next) in
            sum + next.element * weightsHiddenOutput[next.offset]
        }
        
        let output = ActivationFunction.sigmoid(finalOutput + biasesOutput)
        let outputError = expectedOutput - output
        let outputDelta = outputError * ActivationFunction.sigmoidDerivative(output)
        
        let hiddenErrors = weightsHiddenOutput.map { weight in
            outputDelta * weight
        }
        let hiddenDeltas = hiddenErrors.enumerated().map { (i, error) in
            error * ActivationFunction.sigmoidDerivative(hiddenOutputs[i])
        }
        
        // Update weights and biases
        for i in 0..<weightsHiddenOutput.count {
            weightsHiddenOutput[i] += learningRate * outputDelta * hiddenOutputs[i]
        }
        biasesOutput += learningRate * outputDelta
        
        for i in 0..<weightsInputHidden.count {
            for j in 0..<weightsInputHidden[i].count {
                weightsInputHidden[i][j] += learningRate * hiddenDeltas[j] * inputs[i]
            }
            biasesHidden[i] += learningRate * hiddenDeltas[i]
        }
    }
    
    // Train the network
    func train(inputs: [[Double]], targets: [Double], iterations: Int) {
        for _ in 0..<iterations {
            for (input, target) in zip(inputs, targets) {
                forward(input)
                backward(input, target)
            }
        }
    }
    
    // Test the network
    func predict(inputs: [Double]) -> Double {
        return forward(inputs)
    }
}

// Example usage
let nn = NeuralNetwork(inputSize: 3, hiddenSize: 4, learningRate: 0.1)
let trainingInputs = [
    [0.0, 0.0, 1.0],
    [1.0, 1.0, 1.0],
    [1.0, 0.0, 1.0],
    [0.0, 1.0, 1.0]
]
let trainingOutputs = [0.0, 1.0, 1.0, 0.0]
nn.train(inputs: trainingInputs, targets: trainingOutputs, iterations: 10000)

let testOutput = nn.predict(inputs: [1.0, 0.0, 0.0])
print("Output for [1.0, 0.0, 0.0]: \(testOutput)")
