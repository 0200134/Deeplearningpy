import org.deeplearning4j.datasets.iterator.impl.CifarDataSetIterator
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.ConvolutionMode
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers._
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.api.IterationListener
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.learning.config.Adam
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.nd4j.linalg.schedule.{ScheduleType, StepSchedule}
import org.nd4j.evaluation.classification.Evaluation

import java.io.File

object AdvancedCifarClassification {
  def main(args: Array[String]): Unit = {
    val numClasses = 10
    val batchSize = 64
    val epochs = 50

    // Load CIFAR-10 dataset
    val trainData: DataSetIterator = new CifarDataSetIterator(batchSize, numClasses, true)
    val testData: DataSetIterator = new CifarDataSetIterator(batchSize, numClasses, false)

    // Build the sophisticated CNN model
    val model = new MultiLayerNetwork(new NeuralNetConfiguration.Builder()
      .seed(123)
      .weightInit(WeightInit.RELU)
      .updater(new Adam(new StepSchedule(ScheduleType.ITERATION, 0.001, 0.95, 200)))
      .list()
      .layer(new ConvolutionLayer.Builder(5, 5)
        .nIn(3)
        .stride(1, 1)
        .nOut(64)
        .activation(Activation.RELU)
        .convolutionMode(ConvolutionMode.Same)
        .build())
      .layer(new BatchNormalization())
      .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
        .kernelSize(2, 2)
        .stride(2, 2)
        .build())
      .layer(new ConvolutionLayer.Builder(3, 3)
        .nOut(128)
        .stride(1, 1)
        .activation(Activation.RELU)
        .build())
      .layer(new BatchNormalization())
      .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
        .kernelSize(2, 2)
        .stride(2, 2)
        .build())
      .layer(new ConvolutionLayer.Builder(3, 3)
        .nOut(256)
        .stride(1, 1)
        .activation(Activation.RELU)
        .build())
      .layer(new BatchNormalization())
      .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
        .kernelSize(2, 2)
        .stride(2, 2)
        .build())
      .layer(new DropoutLayer.Builder(0.5).build())
      .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
        .nOut(numClasses)
        .activation(Activation.SOFTMAX)
        .build())
      .build())

    model.init()
    model.setListeners(new ScoreIterationListener(100))

    // Train the model
    println("Training model...")
    for (i <- 0 until epochs) {
      model.fit(trainData)
      println(s"Completed epoch ${i + 1}")
    }

    // Evaluate the model
    println("Evaluating model...")
    val eval: Evaluation = model.evaluate(testData)
    println(eval.stats())

    // Save the model
    val modelFile = new File("cifar-cnn-model.zip")
    ModelSerializer.writeModel(model, modelFile, true)

    println("Model training and evaluation complete.")
  }
}
