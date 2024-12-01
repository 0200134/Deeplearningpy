import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.api.IterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Collections;

public class NeuralNetworkExample {

    public static void main(String[] args) throws Exception {

        // Load the training data
        DataSetIterator mnistTrain = new MnistDataSetIterator(64, true, 12345);

        // Normalize the data
        DataNormalization normalizer = new NormalizerStandardize();
        normalizer.fit(mnistTrain);
        mnistTrain.setPreProcessor(normalizer);

        // Build the neural network
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(12345)
                .updater(new Adam(0.001))
                .list()
                .layer(0, new DenseLayer.Builder().nIn(784).nOut(128)
                        .activation(Activation.RELU)
                        .build())
                .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .activation(Activation.SOFTMAX)
                        .nIn(128).nOut(10).build())
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

        // Train the model
        for (int i = 0; i < 10; i++) {
            model.fit(mnistTrain);
        }

        // Evaluate the model
        DataSetIterator mnistTest = new MnistDataSetIterator(64, false, 12345);
        normalizer.fit(mnistTest);
        mnistTest.setPreProcessor(normalizer);

        Evaluation eval = model.evaluate(mnistTest);
        System.out.println(eval.stats());
    }
}
