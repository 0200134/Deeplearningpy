import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.transferlearning.TransferLearningHelper;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.zoo.model.VGG16;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;

public class AdvancedTransferLearning {

    public static void main(String[] args) throws IOException {

        int numClasses = 5;  // Number of output classes for the new task
        int batchSize = 32;
        int numEpochs = 10;
        int seed = 123;

        // Load the pre-trained VGG16 model
        VGG16 vgg16 = (VGG16) VGG16.builder().build();
        MultiLayerNetwork preTrainedModel = (MultiLayerNetwork) vgg16.initPretrained();

        // Fine-tune configuration
        FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder()
                .seed(seed)
                .updater(new Adam(1e-5))
                .weightInit(WeightInit.XAVIER)
                .build();

        // Modify the network architecture for the new task
        MultiLayerNetwork model = new TransferLearning.Builder(preTrainedModel)
                .fineTuneConfiguration(fineTuneConf)
                .setFeatureExtractor("fc2")  // Fine-tune layers after "fc2"
                .removeOutputLayer()
                .addLayer(new DenseLayer.Builder()
                        .activation(Activation.RELU)
                        .nIn(4096)
                        .nOut(1024)
                        .build())
                .addLayer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .activation(Activation.SOFTMAX)
                        .nIn(1024)
                        .nOut(numClasses)
                        .build())
                .build();

        model.init();
        model.setListeners(new ScoreIterationListener(10));

        // Load training and test data
        DataSetIterator trainData = //... load training data
        DataSetIterator testData = //... load test data

        // Normalize the data
        NormalizerStandardize normalizer = new NormalizerStandardize();
        normalizer.fit(trainData);
        trainData.setPreProcessor(normalizer);
        testData.setPreProcessor(normalizer);

        // Train the model
        for (int i = 0; i < numEpochs; i++) {
            model.fit(trainData);
            System.out.println("Completed epoch " + i);
        }

        // Evaluate the model
        Evaluation eval = model.evaluate(testData);
        System.out.println(eval.stats());

        // Save the model
        File modelFile = new File("fine_tuned_vgg16_model.zip");
        ModelSerializer.writeModel(model, modelFile, true);
    }
}
