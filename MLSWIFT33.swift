import TensorFlow

// Define a simple neural network model
struct SimpleModel: Layer {
    var layer1 = Dense<Float>(inputSize: 784, outputSize: 128, activation: relu)
    var layer2 = Dense<Float>(inputSize: 128, outputSize: 64, activation: relu)
    var outputLayer = Dense<Float>(inputSize: 64, outputSize: 10, activation: softmax)

    @differentiable
    func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        let h1 = layer1(input)
        let h2 = layer2(h1)
        return outputLayer(h2)
    }
}

// Load and preprocess the dataset
let (trainImages, trainLabels) = loadMNIST(train: true)
let (testImages, testLabels) = loadMNIST(train: false)

// Initialize the model and optimizer
var model = SimpleModel()
let optimizer = SGD(for: model, learningRate: 0.01)

// Training loop
for epoch in 1...10 {
    for batch in zip(trainImages.batched(32), trainLabels.batched(32)) {
        let (images, labels) = batch
        let gradients = gradient(at: model) { model -> Tensor<Float> in
            let logits = model(images)
            return softmaxCrossEntropy(logits: logits, labels: labels)
        }
        optimizer.update(&model, along: gradients)
    }
    print("Epoch \(epoch) complete")
}

// Evaluate the model
let testAccuracy = accuracy(predicted: model(testImages).argmax(squeezingAxis: 1), expected: testLabels)
print("Test accuracy: \(testAccuracy)")
