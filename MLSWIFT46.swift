import TensorFlow
import PythonKit

let cifar10 = Python.import("tensorflow.keras.datasets.cifar10")

// Load the CIFAR-10 dataset
let (trainData, testData) = cifar10.load_data()
let (trainImages, trainLabels) = (Tensor<Float>(numpy: trainData.0)! / 255.0, Tensor<Int32>(numpy: trainData.1)!)
let (testImages, testLabels) = (Tensor<Float>(numpy: testData.0)! / 255.0, Tensor<Int32>(numpy: testData.1)!)

// Define the CNN model
struct CNN: Layer {
    var conv1 = Conv2D<Float>(filterShape: (3, 3, 3, 32), strides: (1, 1), padding: .same, activation: relu)
    var pool1 = MaxPool2D<Float>(poolSize: (2, 2), strides: (2, 2), padding: .valid)
    var conv2 = Conv2D<Float>(filterShape: (3, 3, 32, 64), strides: (1, 1), padding: .same, activation: relu)
    var pool2 = MaxPool2D<Float>(poolSize: (2, 2), strides: (2, 2), padding: .valid)
    var conv3 = Conv2D<Float>(filterShape: (3, 3, 64, 128), strides: (1, 1), padding: .same, activation: relu)
    var pool3 = MaxPool2D<Float>(poolSize: (2, 2), strides: (2, 2), padding: .valid)
    var flatten = Flatten<Float>()
    var dense1 = Dense<Float>(inputSize: 2048, outputSize: 128, activation: relu)
    var dropout = Dropout<Float>(probability: 0.5)
    var dense2 = Dense<Float>(inputSize: 128, outputSize: 10)
    
    @differentiable
    func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        input.sequenced(through: conv1, pool1, conv2, pool2, conv3, pool3, flatten, dense1, dropout, dense2)
    }
}

// Hyperparameters
let epochCount = 10
let batchSize = 64
let learningRate: Float = 0.001

// Define the model and optimizer
var model = CNN()
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
