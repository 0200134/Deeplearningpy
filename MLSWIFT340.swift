import TensorFlow

struct ConvNet: Layer {
    var conv1 = Conv2D<Float>(filterShape: (5, 5, 1, 32), activation: relu)
    var pool1 = MaxPool2D<Float>(poolSize: (2, 2), strides: (2, 2))
    var conv2 = Conv2D<Float>(filterShape: (5, 5, 32, 64), activation: relu)
    var pool2 = MaxPool2D<Float>(poolSize: (2, 2), strides: (2, 2))
    var flatten = Flatten<Float>()
    var dense1 = Dense<Float>(inputSize: 7*7*64, outputSize: 1024, activation: relu)
    var dropout = Dropout<Float>(probability: 0.5)
    var dense2 = Dense<Float>(inputSize: 1024, outputSize: 10)

    @differentiable
    func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        let x = input.sequenced(through: conv1, pool1, conv2, pool2, flatten, dense1, dropout)
        return dense2(x)
    }
}import Datasets

let mnist = MNIST(flattening: false, normalizing: true)let epochCount = 10
var model = ConvNet()
let optimizer = Adam(for: model)

for epoch in 1...epochCount {
    var totalLoss: Float = 0
    var totalCorrect = 0
    var totalSamples = 0

    for batch in mnist.training {
        let x = batch.data
        let y = batch.label

        let (loss, grad) = model.valueWithGradient { model -> Tensor<Float> in
            let logits = model(x)
            return softmaxCrossEntropy(logits: logits, labels: y)
        }

        optimizer.update(&model, along: grad)
        
        totalLoss += loss.scalarized()
        let logits = model(x)
        let predictions = logits.argmax(squeezingAxis: 1)
        totalCorrect += Tensor<Int32>(predictions .== y).sum().scalarized()
        totalSamples += y.shape[0]
    }
    
    let accuracy = Float(totalCorrect) / Float(totalSamples)
    print("Epoch \(epoch): Loss: \(totalLoss), Accuracy: \(accuracy)")
}
