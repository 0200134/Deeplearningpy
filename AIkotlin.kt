import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.DenseLayer
import org.deeplearning4j.nn.conf.layers.OutputLayer
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.learning.config.Adam
import org.nd4j.linalg.lossfunctions.LossFunctions

fun main() {
    val inputSize = 3072 // 32x32x3 CIFAR-10 images
    val hiddenSize1 = 256
    val hiddenSize2 = 128
    val outputSize = 10

    val model = MultiLayerNetwork(NeuralNetConfiguration.Builder()
        .seed(123)
        .weightInit(WeightInit.XAVIER)
        .updater(Adam(0.001))
        .list()
        .layer(DenseLayer.Builder().nIn(inputSize).nOut(hiddenSize1)
            .activation(Activation.RELU).build())
        .layer(DenseLayer.Builder().nIn(hiddenSize1).nOut(hiddenSize2)
            .activation(Activation.RELU).build())
        .layer(OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
            .activation(Activation.SOFTMAX)
            .nIn(hiddenSize2).nOut(outputSize).build())
        .build())

    model.init()

    // Create dummy input and labels
    val input: INDArray = Nd4j.rand(1, inputSize)
    val labels: INDArray = Nd4j.zeros(1, outputSize)
    labels.putScalar(0, 1)

    val dataSet = DataSet(input, labels)

    // Train the model
    val epochs = 1000
    for (i in 0 until epochs) {
        model.fit(dataSet)
        if (i % 100 == 0) {
            val loss = model.score()
            println("Epoch: $i Loss: $loss")
        }
    }

    println("Training complete.")
}
