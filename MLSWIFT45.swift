import TensorFlow

// Load the MNIST dataset
let (trainImages, trainLabels) = readMNISTDataset(training: true)
let (testImages, testLabels) = readMNISTDataset(training: false)

// Define the neural network model
struct SimpleNN: Layer {
    var layer1 = Dense<Float>(inputSize: 784, outputSize: 128, activation: relu)
    var layer2 = Dense<Float>(inputSize: 128, outputSize: 64, activation: relu)
    var layer3 = Dense<Float>(inputSize: 64, outputSize: 10)

    @differentiable
    func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        return input.sequenced(through: layer1, layer2, layer3)
    }
}

// Hyperparameters
let epochCount = 10
let batchSize = 128
let learningRate: Float = 0.001

// Define the model and optimizer
var model = SimpleNN()
let optimizer = Adam(for: model, learningRate: learningRate)

// Training loop
for epoch in 1...epochCount {
    let trainingShuffled = zip(trainImages.shuffled(), trainLabels.shuffled())
    for batch in trainingShuffled.batched(batchSize) {
        let (batchImages, batchLabels) = (batch.0, batch.1)
        let (loss, grad) = model.valueWithGradient { model -> Tensor<Float> in
            let logits = model(batchImages)
            return softmaxCrossEntropy(logits: logits, labels: batchLabels)
        }
        optimizer.update(&model, along: grad)
    }
    print("Epoch \(epoch) completed")
}

// Evaluate the model
let testPredictions = model(testImages).argmax(squeezingAxis: 1)
let accuracy = TensorFlow.metrics.accuracy(predicted: testPredictions, expected: testLabels)
print("Test accuracy: \(accuracy)")

// Helper function to read MNIST dataset
func readMNISTDataset(training: Bool) -> (Tensor<Float>, Tensor<Int32>) {
    let prefix = training ? "train" : "t10k"
    let images = Tensor<Float>(numpy: loadMNISTImages("\(prefix)-images-idx3-ubyte"))
    let labels = Tensor<Int32>(numpy: loadMNISTLabels("\(prefix)-labels-idx1-ubyte"))
    return (images.reshaped(to: [-1, 28 * 28]) / 255.0, labels)
}

// Helper functions to load MNIST data from files (not shown here)
// ...
