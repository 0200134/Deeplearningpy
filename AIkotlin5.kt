import org.deeplearning4j.nn.conf.MultiLayerConfiguration
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.DenseLayer
import org.deeplearning4j.nn.conf.layers.OutputLayer
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.nd4j.linalg.factory.Nd4j

fun main() {
    // Number of input nodes (28x28 pixels) and output nodes (10 classes)
    val inputSize = 784
    val outputSize = 10
    val hiddenNodes = 1000

    // Build the configuration for the neural network
    val conf: MultiLayerConfiguration = NeuralNetConfiguration.Builder()
        .seed(123)  // Random seed for reproducibility
        .weightInit(WeightInit.XAVIER)
        .activation(Activation.RELU)
        .list()
        .layer(0, DenseLayer.Builder().nIn(inputSize).nOut(hiddenNodes).build())
        .layer(1, DenseLayer.Builder().nIn(hiddenNodes).nOut(hiddenNodes).build())
        .layer(2, OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
            .activation(Activation.SOFTMAX)
            .nIn(hiddenNodes).nOut(outputSize).build())
        .build()

    // Initialize and configure the network
    val model = MultiLayerNetwork(conf)
    model.init()
    model.setListeners(ScoreIterationListener(100))

    // Load and normalize the MNIST data
    val trainIter: DataSetIterator = MnistDataSetIterator(64, true, 12345)
    val testIter: DataSetIterator = MnistDataSetIterator(64, false, 12345)
    val normalizer: DataNormalization = NormalizerMinMaxScaler(0.0, 1.0)
    normalizer.fit(trainIter)
    trainIter.setPreProcessor(normalizer)
    testIter.setPreProcessor(normalizer)

    // Train the model
    for (epoch in 0 until 10) {
        model.fit(trainIter)
        println("Epoch $epoch complete.")
    }

    // Evaluate the model on test data
    val eval = model.evaluate(testIter)
    println(eval.stats())
}
