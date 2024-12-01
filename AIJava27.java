import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimizers.optimizers.SGD;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

public class HandwrittenDigitRecognition {
    public static void main(String[] args) throws Exception {
        // Load the MNIST dataset
        DataSetIterator mnistTrain = new MnistDataSetIterator(60000, true, 12345);
        DataSetIterator mnistTest = new MnistDataSetIterator(10000, false, 12345);

        // Create a simple multilayer perceptron
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(12345)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new SGD())
                .l2(0.0001)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(784).nOut(100).activation(Activation.RELU).build())
                .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .activation(Activation.SOFTMAX)
                        .nIn(100).nOut(10).build())
                .pretrain(false).backprop(true).build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

        // Train the model
        for (int i = 0; i < 10; i++) {
            model.fit(mnistTrain);
        }

        // Evaluate the model on the test set
        Evaluation eval = new Evaluation(10);
        model.evaluate(mnistTest, eval);
        System.out.println(eval.stats());
    }
}
