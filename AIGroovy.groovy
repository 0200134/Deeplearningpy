import org.deeplearning4j.datasets.iterator.impl.CifarDataSetIterator
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.ConvolutionMode
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.conf.layers.*
import org.deeplearning4j.nn.graph.ComputationGraph
import org.deeplearning4j.nn.graph.GraphBuilder
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.api.IterationListener
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.util.ModelSerializer
import org.deeplearning4j.zoo.model.ResNet50
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.learning.config.Adam
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.nd4j.linalg.schedule.ScheduleType
import org.nd4j.linalg.schedule.StepSchedule
import org.nd4j.evaluation.classification.Evaluation
import org.datavec.api.split.FileSplit
import org.datavec.image.loader.NativeImageLoader
import org.datavec.image.recordreader.ImageRecordReader
import org.datavec.image.transform.*
import java.io.File
import java.util.Arrays
import java.util.Random

class AdvancedCifarClassification {
    static void main(String[] args) {
        int numClasses = 10
        int batchSize = 64
        int epochs = 50
        long seed = 123

        // Image augmentation
        ImageTransform transform = new PipelineImageTransform.Builder()
                .addImageTransform(new ScaleImageTransform(0.2f))
                .addImageTransform(new WarpImageTransform(0.1f))
                .addImageTransform(new FlipImageTransform(1))
                .build()

        // Load CIFAR-10 dataset with augmentation
        DataSetIterator trainData = new CifarDataSetIterator(batchSize, numClasses, true, transform)
        DataSetIterator testData = new CifarDataSetIterator(batchSize, numClasses, false)

        // Build the sophisticated CNN model
        ComputationGraph model = new ComputationGraph(new NeuralNetConfiguration.Builder()
                .seed(seed)
                .weightInit(WeightInit.RELU)
                .updater(new Adam(new StepSchedule(ScheduleType.ITERATION, 0.001, 0.95, 200)))
                .graphBuilder()
                .addInputs("input")
                .setInputTypes(InputType.convolutional(32, 32, 3))
                .addLayer("conv1", new ConvolutionLayer.Builder(5, 5)
                        .nIn(3)
                        .stride(1, 1)
                        .nOut(64)
                        .activation(Activation.RELU)
                        .convolutionMode(ConvolutionMode.Same)
                        .build(), "input")
                .addLayer("batchnorm1", new BatchNormalization(), "conv1")
                .addLayer("pool1", new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build(), "batchnorm1")
                .addLayer("conv2", new ConvolutionLayer.Builder(3, 3)
                        .nOut(128)
                        .stride(1, 1)
                        .activation(Activation.RELU)
                        .build(), "pool1")
                .addLayer("batchnorm2", new BatchNormalization(), "conv2")
                .addLayer("pool2", new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build(), "batchnorm2")
                .addLayer("conv3", new ConvolutionLayer.Builder(3, 3)
                        .nOut(256)
                        .stride(1, 1)
                        .activation(Activation.RELU)
                        .build(), "pool2")
                .addLayer("batchnorm3", new BatchNormalization(), "conv3")
                .addLayer("pool3", new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build(), "batchnorm3")
                .addLayer("dropout", new DropoutLayer.Builder(0.5).build(), "pool3")
                .addLayer("output", new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(numClasses)
                        .activation(Activation.SOFTMAX)
                        .build(), "dropout")
                .setOutputs("output")
                .build())

        model.init()
        model.setListeners(new ScoreIterationListener(100))

        // Train the model
        println("Training model...")
        for (int i = 0; i < epochs; i++) {
            model.fit(trainData)
            println("Completed epoch " + (i + 1))
        }

        // Evaluate the model
        println("Evaluating model...")
        Evaluation eval = model.evaluate(testData)
        println(eval.stats())

        // Save the model
        File modelFile = new File("cifar-cnn-model.zip")
        ModelSerializer.writeModel(model, modelFile, true)

        println("Model training and evaluation complete.")
    }
}
