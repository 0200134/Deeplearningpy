import TensorFlow
import PythonKit

let cifar10 = Python.import("tensorflow.keras.datasets.cifar10")
let (trainData, testData) = cifar10.load_data()
let (trainImages, trainLabels) = (Tensor<Float>(numpy: trainData.0)! / 255.0, Tensor<Int32>(numpy: trainData.1)!)
let (testImages, testLabels) = (Tensor<Float>(numpy: testData.0)! / 255.0, Tensor<Int32>(numpy: testData.1)!)

// Data augmentation
func augment(images: Tensor<Float>) -> Tensor<Float> {
    return images
        .randomCrop(size: [32, 32, 3])
        .flippedHorizontally()
        .randomContrast(factor: 0.2)
        .randomBrightness(factor: 0.2)
}

// Define a Residual Block
struct ResidualBlock: Layer {
    var conv1: Conv2D<Float>
    var conv2: Conv2D<Float>
    var batchNorm1: BatchNorm<Float>
    var batchNorm2: BatchNorm<Float>
    @noDerivative let isShortcutNeeded: Bool
    @noDerivative let shortcutConv: Conv2D<Float>?
    
    init(filters: (Int, Int), strides: (Int, Int) = (1, 1)) {
        conv1 = Conv2D<Float>(filterShape: (3, 3, filters.0, filters.1), strides: strides, padding: .same)
        batchNorm1 = BatchNorm<Float>(featureCount: filters.1)
        conv2 = Conv2D<Float>(filterShape: (3, 3, filters.1, filters.1), padding: .same)
        batchNorm2 = BatchNorm<Float>(featureCount: filters.1)
        
        if filters.0 != filters.1 || strides != (1, 1) {
            isShortcutNeeded = true
            shortcutConv = Conv2D<Float>(filterShape: (1, 1, filters.0, filters.1), strides: strides)
        } else {
            isShortcutNeeded = false
            shortcutConv = nil
        }
    }
    
    @differentiable
    func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        let shortcut: Tensor<Float>
        if isShortcutNeeded {
            shortcut = shortcutConv!(input)
        } else {
            shortcut = input
        }
        
        var out = input.sequenced(through: conv1, batchNorm1, relu)
        out = out.sequenced(through: conv2, batchNorm2)
        return relu(out + shortcut)
    }
}

// Define ResNet
struct ResNet: Layer {
    var conv1 = Conv2D<Float>(filterShape: (3, 3, 3, 64), padding: .same)
    var batchNorm1 = BatchNorm<Float>(featureCount: 64)
    var layer1 = Sequential {
        ResidualBlock(filters: (64, 64))
        ResidualBlock(filters: (64, 64))
    }
    var layer2 = Sequential {
        ResidualBlock(filters: (64, 128), strides: (2, 2))
        ResidualBlock(filters: (128, 128))
    }
    var layer3 = Sequential {
        ResidualBlock(filters: (128, 256), strides: (2, 2))
        ResidualBlock(filters: (256, 256))
    }
    var globalAvgPool = GlobalAvgPool2D<Float>()
    var flatten = Flatten<Float>()
    var dense = Dense<Float>(inputSize: 256, outputSize: 10)
    
    @differentiable
    func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        return input.sequenced(through: conv1, batchNorm1, relu, layer1, layer2, layer3, globalAvgPool, flatten, dense)
    }
}

// Hyperparameters
let epochCount = 200
let batchSize = 64
let learningRate: Float = 0.001

// Define the model and optimizer
var model = ResNet()
let optimizer = Adam(for: model, learningRate: learningRate)

// Training loop
for epoch in 1...epochCount {
    let augmentedImages = augment(images: trainImages)
    let trainingShuffled = zip(augmentedImages.shuffled(), trainLabels.shuffled())
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
