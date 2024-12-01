import TensorFlow

// Define a simple model
struct Model: Layer {
    var layer1 = Dense<Float>(inputSize: 784, outputSize: 128, activation: relu)
    var layer2 = Dense<Float>(inputSize: 128, outputSize: 10, activation: softmax)

    @differentiable
    func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        return input.sequenced(through: layer1, layer2)
    }
}

// Load data (example using random data)
let batchSize = 64
let input = Tensor<Float>(randomNormal: [batchSize, 784])
let labels = Tensor<Int32>(randomUniform: [batchSize], maxValue: 10)

// Initialize model and optimizer
var model = Model()
let optimizer = SGD(for: model, learningRate: 0.01)

// Training loop (simplified)
for epoch in 1...10 {
    let (loss, grad) = valueWithGradient(at: model) { model -> Tensor<Float> in
        let logits = model(input)
        return softmaxCrossEntropy(logits: logits, labels: labels)
    }
    optimizer.update(&model, along: grad)
    print("Epoch \(epoch): Loss: \(loss)")
}
