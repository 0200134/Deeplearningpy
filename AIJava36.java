import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;

public class SimpleNeuralNetwork {

    public static void main(String[] args) {

        int inputSize = 10;   // Number of input features
        int outputSize = 2;   // Number of output classes

        // Configure the neural network
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(123)    // Random seed for reproducibility
                .updater(new Adam(0.001))   // Adam optimizer
                .list()
                .layer(0, new DenseLayer.Builder().nIn(inputSize).nOut(50)
                        .activation(Activation.RELU)
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .layer(1, new DenseLayer.Builder().nIn(50).nOut(50)
                        .activation(Activation.RELU)
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .layer(2, new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD)
                        .activation(Activation.SOFTMAX)
                        .nIn(50).nOut(outputSize).build())
                .backpropType(BackpropType.Standard)
                .build();

        // Build and initialize the network
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(10));

        // Load your data here
        DataSetIterator trainData = //... load data here
        DataSetIterator testData = //... load data here

        // Normalize the data
        NormalizerStandardize normalizer = new NormalizerStandardize();
        normalizer.fit(trainData);
        trainData.setPreProcessor(normalizer);
        testData.setPreProcessor(normalizer);

        // Train the model
        int numEpochs = 50;
        model.fit(trainData, numEpochs);

        // Evaluate the model on test data
        Evaluation eval = model.evaluate(testData);
        System.out.println(eval.stats());
    }
}
