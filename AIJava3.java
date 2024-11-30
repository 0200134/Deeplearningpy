import org.deeplearning4j.datasets.iterator.impl.CifarDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.BatchNormalization;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DropoutLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.schedule.ScheduleType;
import org.nd4j.linalg.schedule.StepSchedule;

import java.io.File;
import java.io.IOException;

public class AdvancedCifarClassification {
    public static void main(String[] args) throws IOException {
        int numClasses = 10;
        int batchSize = 64;
        int epochs = 50;

        // Load CIFAR-10 dataset
        DataSetIterator trainData = new CifarDataSetIterator(batchSize, numClasses, true);
        DataSetIterator testData = new CifarDataSetIterator(batchSize, numClasses, false);

        // Build the sophisticated CNN model
        MultiLayerNetwork model = new MultiLayerNetwork(new NeuralNetConfiguration.Builder()
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
            .build());

        model.init();
        model.setListeners(new ScoreIterationListener(100));

        // Train the model
        System.out.println("Training model...");
        for (int i = 0; i < epochs; i++) {
            model.fit(trainData);
            System.out.println("Completed epoch " + (i+1));
        }

        // Evaluate the model
        System.out.println("Evaluating model...");
        Evaluation eval = model.evaluate(testData);
        System.out.println(eval.stats());

        // Save the model
        File modelFile = new File("cifar-cnn-model.zip");
        ModelSerializer.writeModel(model, modelFile, true);

        System.out.println("Model training and evaluation complete.");
    }
}
