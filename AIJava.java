import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class NeuralNetwork {
    public static void main(String[] args) {
        int inputSize = 3072; // 32x32x3 CIFAR-10 images
        int hiddenSize1 = 256;
        int hiddenSize2 = 128;
        int outputSize = 10;

        MultiLayerNetwork model = new MultiLayerNetwork(new NeuralNetConfiguration.Builder()
            .seed(123)
            .weightInit(WeightInit.XAVIER)
            .updater(new Adam(0.001))
            .list()
            .layer(new DenseLayer.Builder().nIn(inputSize).nOut(hiddenSize1)
                    .activation(Activation.RELU).build())
            .layer(new DenseLayer.Builder().nIn(hiddenSize1).nOut(hiddenSize2)
                    .activation(Activation.RELU).build())
            .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                    .activation(Activation.SOFTMAX)
                    .nIn(hiddenSize2).nOut(outputSize).build())
            .build());

        model.init();

        // Create dummy input and labels
        INDArray input = Nd4j.rand(1, inputSize);
        INDArray labels = Nd4j.zeros(1, outputSize);
        labels.putScalar(0, 1);

        DataSet dataSet = new DataSet(input, labels);

        // Train the model
        int epochs = 1000;
        for (int i = 0; i < epochs; i++) {
            model.fit(dataSet);
            if (i % 100 == 0) {
                double loss = model.score();
                System.out.println("Epoch: " + i + " Loss: " + loss);
            }
        }

        System.out.println("Training complete.");
    }
}
