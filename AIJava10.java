import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.GRU;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.datasets.iterator.impl.SentimentExampleIterator;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class GRUExample {

    public static void main(String[] args) throws Exception {

        int batchSize = 64;
        int epochs = 10;
        int vectorSize = 300; // Size of the word vectors

        // Load the sentiment analysis dataset
        DataSetIterator trainData = new SentimentExampleIterator(batchSize, "train", vectorSize);
        DataSetIterator testData = new SentimentExampleIterator(batchSize, "test", vectorSize);

        // Build the GRU network
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(123)
                .updater(new Adam(1e-3))
                .list()
                .layer(new GRU.Builder().nIn(vectorSize).nOut(256)
                        .activation(Activation.TANH).build())
                .layer(new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX).nIn(256).nOut(2).build())
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(10));

        // Train the model
        System.out.println("Training model...");
        for (int i = 0; i < epochs; i++) {
            model.fit(trainData);
            System.out.println("Epoch " + i + " complete. Evaluating...");
            Evaluation eval = model.evaluate(testData);
            System.out.println(eval.stats());
            testData.reset();
        }

        System.out.println("Model training complete.");
    }
}
