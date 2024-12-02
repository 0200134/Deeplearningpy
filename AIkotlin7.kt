import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration
import org.deeplearning4j.earlystopping.termination.EpochTerminationCondition
import org.deeplearning4j.earlystopping.termination.ScoreImprovementEpochTerminationCondition
import org.deeplearning4j.earlystopping.saver.LocalFileModelSaver
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingTrainer
import org.deeplearning4j.nn.conf.MultiLayerConfiguration
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.DenseLayer
import org.deeplearning4j.nn.conf.layers.DropoutLayer
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
    val inputSize = 784
    val outputSize = 10
    val hiddenNodes = 1000

    val conf: MultiLayerConfiguration = NeuralNetConfiguration.Builder()
        .seed(123)
        .weightInit(WeightInit.XAVIER)
        .activation(Activation.RELU)
        .list()
        .layer(0, DenseLayer.Builder().nIn(inputSize).nOut(hiddenNodes).build())
        .layer(1, DropoutLayer.Builder(0.5).build())
        .layer(2, DenseLayer.Builder().nIn(hiddenNodes).nOut(hiddenNodes).build())
        .layer(3, DenseLayer.Builder().nIn(hiddenNodes).nOut(hiddenNodes).build())
        .layer(4, DropoutLayer.Builder(0.5).build())
        .layer(5, OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
            .activation(Activation.SOFTMAX)
            .nIn(hiddenNodes).nOut(outputSize).build())
        .build()

    val model = MultiLayerNetwork(conf)
    model.init()
    model.setListeners(ScoreIterationListener(100))

    val trainIter: DataSetIterator = MnistDataSetIterator(64, true, 12345)
    val testIter: DataSetIterator = MnistDataSetIterator(64, false, 12345)
    val normalizer: DataNormalization = NormalizerMinMaxScaler(0.0, 1.0)
    normalizer.fit(trainIter)
    trainIter.setPreProcessor(normalizer)
    testIter.setPreProcessor(normalizer)

    val saver = LocalFileModelSaver("models/")
    val earlyStoppingConfig = EarlyStoppingConfiguration.Builder<MultiLayerNetwork>()
        .epochTerminationConditions(ScoreImprovementEpochTerminationCondition(5))
        .evaluateEveryNEpochs(1)
        .modelSaver(saver)
        .build()

    val trainer = EarlyStoppingTrainer(earlyStoppingConfig, model, trainIter)

    val result = trainer.fit()
    println("Termination reason: ${result.terminationReason}")
    println("Termination details: ${result.terminationDetails}")
    println("Total epochs: ${result.totalEpochs}")
    println("Best epoch number: ${result.bestModelEpoch}")
    println("Score at best epoch: ${result.bestModelScore}")

    val bestModel = result.bestModel
    val eval = bestModel.evaluate(testIter)
    println(eval.stats())
}
