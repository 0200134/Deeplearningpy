import TensorFlow

struct MultiHeadAttention: Layer {
    var queryLayer: Dense<Float>
    var keyLayer: Dense<Float>
    var valueLayer: Dense<Float>
    var outputLayer: Dense<Float>
    var numHeads: Int

    init(dModel: Int, numHeads: Int) {
        self.numHeads = numHeads
        self.queryLayer = Dense<Float>(inputSize: dModel, outputSize: dModel)
        self.keyLayer = Dense<Float>(inputSize: dModel, outputSize: dModel)
        self.valueLayer = Dense<Float>(inputSize: dModel, outputSize: dModel)
        self.outputLayer = Dense<Float>(inputSize: dModel, outputSize: dModel)
    }

    @differentiable
    func scaledDotProductAttention(query: Tensor<Float>, key: Tensor<Float>, value: Tensor<Float>) -> Tensor<Float> {
        let matmulQK = matmul(query, key.transposed(permutation: [0, 2, 1]))
        let dk = Tensor<Float>(key.shape.last!)
        let scaledAttentionLogits = matmulQK / sqrt(dk)
        let attentionWeights = softmax(scaledAttentionLogits, axis: -1)
        let output = matmul(attentionWeights, value)
        return output
    }

    @differentiable
    func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        let query = queryLayer(input)
        let key = keyLayer(input)
        let value = valueLayer(input)

        let batchSize = input.shape[0]
        let seqLength = input.shape[1]
        let depth = query.shape[2] / numHeads

        let queryReshaped = query.reshaped(to: [batchSize, seqLength, numHeads, depth]).transposed(permutation: [0, 2, 1, 3])
        let keyReshaped = key.reshaped(to: [batchSize, seqLength, numHeads, depth]).transposed(permutation: [0, 2, 1, 3])
        let valueReshaped = value.reshaped(to: [batchSize, seqLength, numHeads, depth]).transposed(permutation: [0, 2, 1, 3])

        let attentionOutput = scaledDotProductAttention(query: queryReshaped, key: keyReshaped, value: valueReshaped)

        let attentionOutputReshaped = attentionOutput.transposed(permutation: [0, 2, 1, 3]).reshaped(to: [batchSize, seqLength, depth * numHeads])

        return outputLayer(attentionOutputReshaped)
    }
}

struct TransformerEncoderLayer: Layer {
    var multiHeadAttention: MultiHeadAttention
    var dense1: Dense<Float>
    var dense2: Dense<Float>
    var layerNorm1: LayerNorm<Float>
    var layerNorm2: LayerNorm<Float>
    var dropout: Dropout<Float>

    init(dModel: Int, numHeads: Int, dff: Int, rate: Float = 0.1) {
        self.multiHeadAttention = MultiHeadAttention(dModel: dModel, numHeads: numHeads)
        self.dense1 = Dense<Float>(inputSize: dModel, outputSize: dff, activation: relu)
        self.dense2 = Dense<Float>(inputSize: dff, outputSize: dModel)
        self.layerNorm1 = LayerNorm(featureCount: dModel)
        self.layerNorm2 = LayerNorm(featureCount: dModel)
        self.dropout = Dropout(probability: rate)
    }

    @differentiable
    func callAsFunction(_ input: Tensor<Float>, training: Bool = false) -> Tensor<Float> {
        let attentionOutput = multiHeadAttention(input)
        let attentionOutputDropout = dropout(attentionOutput, during: training)
        let out1 = layerNorm1(input + attentionOutputDropout)
        let denseOutput = dense2(dense1(out1))
        let denseOutputDropout = dropout(denseOutput, during: training)
        let out2 = layerNorm2(out1 + denseOutputDropout)
        return out2
    }
}

struct TransformerEncoder: Layer {
    var layers: [TransformerEncoderLayer]
    var numLayers: Int

    init(numLayers: Int, dModel: Int, numHeads: Int, dff: Int, rate: Float = 0.1) {
        self.numLayers = numLayers
        self.layers = (0..<numLayers).map { _ in
            TransformerEncoderLayer(dModel: dModel, numHeads: numHeads, dff: dff, rate: rate)
        }
    }

    @differentiable
    func callAsFunction(_ input: Tensor<Float>, training: Bool = false) -> Tensor<Float> {
        var x = input
        for layer in layers {
            x = layer(x, training: training)
        }
        return x
    }
}

// Usage example
let dModel = 512
let numHeads = 8
let dff = 2048
let numLayers = 6
let encoder = TransformerEncoder(numLayers: numLayers, dModel: dModel, numHeads: numHeads, dff: dff)

let exampleInput = Tensor<Float>(randomUniform: [1, 10, dModel])
let output = encoder(exampleInput)
print(output)
