import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.dataset.DataSet;

public class GANExample {

    public static void main(String[] args) throws Exception {
        int noiseDim = 100;
        int batchSize = 64;
        int epochs = 10000;

        // Build Generator Model
        MultiLayerConfiguration genConf = new NeuralNetConfiguration.Builder()
                .updater(new Adam(0.0002, 0.5))
                .list()
                .layer(new DenseLayer.Builder().nIn(noiseDim).nOut(256).activation(Activation.RELU).build())
                .layer(new DenseLayer.Builder().nIn(256).nOut(512).activation(Activation.RELU).build())
                .layer(new DenseLayer.Builder().nIn(512).nOut(1024).activation(Activation.RELU).build())
                .layer(new OutputLayer.Builder().activation(Activation.TANH).nIn(1024).nOut(784).build())
                .build();

        MultiLayerNetwork generator = new MultiLayerNetwork(genConf);
        generator.init();

        // Build Discriminator Model
        MultiLayerConfiguration discConf = new NeuralNetConfiguration.Builder()
                .updater(new Adam(0.0002, 0.5))
                .list()
                .layer(new DenseLayer.Builder().nIn(784).nOut(1024).activation(Activation.LEAKYRELU).build())
                .layer(new DenseLayer.Builder().nIn(1024).nOut(512).activation(Activation.LEAKYRELU).build())
                .layer(new DenseLayer.Builder().nIn(512).nOut(256).activation(Activation.LEAKYRELU).build())
                .layer(new OutputLayer.Builder().activation(Activation.SIGMOID).nIn(256).nOut(1).build())
                .build();

        MultiLayerNetwork discriminator = new MultiLayerNetwork(discConf);
        discriminator.init();

        // Training the GAN
        for (int epoch = 0; epoch < epochs; epoch++) {
            // Train discriminator with real data
            INDArray realData = Nd4j.rand(new int[]{batchSize, 784});
            INDArray realLabels = Nd4j.ones(batchSize);
            DataSet realDataSet = new DataSet(realData, realLabels);
            discriminator.fit(realDataSet);

            // Train discriminator with fake data
            INDArray noise = Nd4j.randn(new int[]{batchSize, noiseDim});
            INDArray fakeData = generator.output(noise);
            INDArray fakeLabels = Nd4j.zeros(batchSize);
            DataSet fakeDataSet = new DataSet(fakeData, fakeLabels);
            discriminator.fit(fakeDataSet);

            // Train generator
            INDArray misleadingLabels = Nd4j.ones(batchSize);
            DataSet misleadingDataSet = new DataSet(noise, misleadingLabels);
            generator.fit(misleadingDataSet);

            if (epoch % 100 == 0) {
                System.out.println("Epoch " + epoch + " complete.");
            }
        }

        System.out.println("Model training complete.");
    }
}
