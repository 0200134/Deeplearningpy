import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.impl.ListDataSetIterator;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.Collections;

public class VAEExample {

    public static void main(String[] args) throws Exception {
        int inputSize = 28 * 28; // MNIST data
        int latentSize = 2;      // Latent space size
        int batchSize = 128;
        int numEpochs = 50;

        // Build the encoder
        MultiLayerConfiguration encoderConf = new NeuralNetConfiguration.Builder()
                .seed(123)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(0.001))
                .list()
                .layer(0, new DenseLayer.Builder().nIn(inputSize).nOut(512)
                        .activation(Activation.RELU).build())
                .layer(1, new DenseLayer.Builder().nOut(256)
                        .activation(Activation.RELU).build())
                .layer(2, new DenseLayer.Builder().nOut(latentSize)
                        .activation(Activation.IDENTITY).build())
                .build();

        // Build the decoder
        MultiLayerConfiguration decoderConf = new NeuralNetConfiguration.Builder()
                .seed(123)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(0.001))
                .list()
                .layer(0, new DenseLayer.Builder().nIn(latentSize).nOut(256)
                        .activation(Activation.RELU).build())
                .layer(1, new DenseLayer.Builder().nOut(512)
                        .activation(Activation.RELU).build())
                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .activation(Activation.SIGMOID)
                        .nOut(inputSize)
                        .build())
                .build();

        MultiLayerNetwork encoder = new MultiLayerNetwork(encoderConf);
        MultiLayerNetwork decoder = new MultiLayerNetwork(decoderConf);
        encoder.init();
        decoder.init();

        encoder.setListeners(Collections.singletonList(new ScoreIterationListener(10)));
        decoder.setListeners(Collections.singletonList(new ScoreIterationListener(10)));

        // Prepare training data
        DataSetIterator mnist = new MnistDataSetIterator(batchSize, true, 12345);

        for (int i = 0; i < numEpochs; i++) {
            DataSet dataSet = mnist.next();
            INDArray input = dataSet.getFeatures();
            INDArray encoded = encoder.output(input);
            INDArray decoded = decoder.output(encoded);

            encoder.fit(input, encoded);
            decoder.fit(encoded, input);

            if (i % 10 == 0) {
                System.out.println("Epoch " + i);
                System.out.println("Encoder Loss: " + encoder.score());
                System.out.println("Decoder Loss: " + decoder.score());
            }
        }

        // Save the models
        ModelSerializer.writeModel(encoder, "VAEEncoderModel.zip", true);
        ModelSerializer.writeModel(decoder, "VAEDecoderModel.zip", true);
    }
}
