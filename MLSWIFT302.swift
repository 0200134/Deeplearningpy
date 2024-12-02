import TensorFlow

struct RNN: Layer {
    var rnn = RNN(LSTMCell<Float>(inputSize: 28, hiddenSize: 128))
    var dense = Dense<Float>(inputSize: 128, outputSize: 10)

    @differentiable
    func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        let rnnOutput = rnn(input).lastOutput
        return dense(rnnOutput)
    }
}let epochCount = 10
var model = RNN()
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
