import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.Collections;

public class GANExample {

    public static void main(String[] args) throws Exception {
        int seed = 12345;
        int epochs = 10000;
        int batchSize = 128;
        int latentSize = 100;
        int numOutputs = 28 * 28; // For MNIST data

        MultiLayerConfiguration generatorConf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(1e-4))
                .list()
                .layer(0, new DenseLayer.Builder().nIn(latentSize).nOut(256)
                        .activation(Activation.RELU).build())
                .layer(1, new DenseLayer.Builder().nOut(512)
                        .activation(Activation.RELU).build())
                .layer(2, new DenseLayer.Builder().nOut(1024)
                        .activation(Activation.RELU).build())
                .layer(3, new DenseLayer.Builder().nOut(numOutputs)
                        .activation(Activation.TANH).build())
                .build();

        MultiLayerConfiguration discriminatorConf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(1e-4))
                .list()
                .layer(0, new DenseLayer.Builder().nIn(numOutputs).nOut(1024)
                        .activation(Activation.LEAKYRELU).build())
                .layer(1, new DenseLayer.Builder().nOut(512)
                        .activation(Activation.LEAKYRELU).build())
                .layer(2, new DenseLayer.Builder().nOut(256)
                        .activation(Activation.LEAKYRELU).build())
                .layer(3, new DenseLayer.Builder().nOut(1)
                        .activation(Activation.SIGMOID).build())
                .build();

        MultiLayerNetwork generator = new MultiLayerNetwork(generatorConf);
        MultiLayerNetwork discriminator = new MultiLayerNetwork(discriminatorConf);
        generator.init();
        discriminator.init();

        generator.setListeners(Collections.singletonList(new ScoreIterationListener(100)));
        discriminator.setListeners(Collections.singletonList(new ScoreIterationListener(100)));

        // Prepare training data
        DataSetIterator mnist = new MnistDataSetIterator(batchSize, true, seed);

        for (int i = 0; i < epochs; i++) {
            DataSet realBatch = mnist.next();
            INDArray realImages = realBatch.getFeatures();
            INDArray realLabels = Nd4j.ones(batchSize, 1);

            INDArray noise = Nd4j.randn(batchSize, latentSize);
            INDArray fakeImages = generator.output(noise);
            INDArray fakeLabels = Nd4j.zeros(batchSize, 1);

            INDArray dInput = Nd4j.vstack(realImages, fakeImages);
            INDArray dLabels = Nd4j.vstack(realLabels, fakeLabels);

            discriminator.fit(dInput, dLabels);

            INDArray trickLabels = Nd4j.ones(batchSize, 1);
            discriminator.output(noise);

            generator.fit(noise, trickLabels);

            if (i % 1000 == 0) {
                System.out.println("Epoch " + i);
                System.out.println("Discriminator Loss: " + discriminator.score());
                System.out.println("Generator Loss: " + generator.score());
            }
        }
    }
}
