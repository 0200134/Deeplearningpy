import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.api.IterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.nd4j.linalg.dataset.api.iterator.SeqDataSetIterator;

import java.util.Arrays;

public class LSTMExample {

    public static void main(String[] args) throws Exception {

        // Sequence data
        INDArray inputData = Nd4j.create(new double[][]{
            {1, 2, 3, 4, 5},
            {2, 3, 4, 5, 6},
            {3, 4, 5, 6, 7},
            {4, 5, 6, 7, 8}
        });

        INDArray labelsData = Nd4j.create(new double[][]{
            {6},
            {7},
            {8},
            {9}
        });

        DataSet dataSet = new DataSet(inputData, labelsData);
        DataSetIterator iterator = new SeqDataSetIterator(Arrays.asList(dataSet), 1);

        // Build the LSTM network
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .updater(new Adam(0.01))
                .list()
                .layer(new LSTM.Builder().nIn(5).nOut(50)
                        .activation(Activation.TANH).build())
                .layer(new RnnOutputLayer.Builder(LossFunction.MSE)
                        .activation(Activation.IDENTITY).nIn(50).nOut(1).build())
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

        // Train the model
        for (int i = 0; i < 100; i++) {
            iterator.reset();
            model.fit(iterator);
        }

        // Test the model
        INDArray testInput = Nd4j.create(new double[][]{{5, 6, 7, 8, 9}});
        INDArray testOutput = model.rnnTimeStep(testInput);
        System.out.println("Prediction: " + testOutput);

    }
}
