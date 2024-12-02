import java.util.Random;

public class NeuralNetwork {
    private double[][] weightsInputHidden;
    private double[][] weightsHiddenOutput;
    private double[] hiddenLayer;
    private double[] outputLayer;
    private double learningRate;

    public NeuralNetwork(int inputSize, int hiddenSize, int outputSize, double learningRate) {
        weightsInputHidden = new double[inputSize][hiddenSize];
        weightsHiddenOutput = new double[hiddenSize][outputSize];
        hiddenLayer = new double[hiddenSize];
        outputLayer = new double[outputSize];
        this.learningRate = learningRate;
        initializeWeights();
    }

    private void initializeWeights() {
        Random rand = new Random();
        for (int i = 0; i < weightsInputHidden.length; i++) {
            for (int j = 0; j < weightsInputHidden[i].length; j++) {
                weightsInputHidden[i][j] = rand.nextGaussian();
            }
        }
        for (int i = 0; i < weightsHiddenOutput.length; i++) {
            for (int j = 0; j < weightsHiddenOutput[i].length; j++) {
                weightsHiddenOutput[i][j] = rand.nextGaussian();
            }
        }
    }

    private double[] sigmoid(double[] layer) {
        double[] result = new double[layer.length];
        for (int i = 0; i < layer.length; i++) {
            result[i] = 1 / (1 + Math.exp(-layer[i]));
        }
        return result;
    }

    public double[] feedForward(double[] inputs) {
        for (int i = 0; i < hiddenLayer.length; i++) {
            hiddenLayer[i] = 0;
            for (int j = 0; j < inputs.length; j++) {
                hiddenLayer[i] += inputs[j] * weightsInputHidden[j][i];
            }
        }
        hiddenLayer = sigmoid(hiddenLayer);

        for (int i = 0; i < outputLayer.length; i++) {
            outputLayer[i] = 0;
            for (int j = 0; j < hiddenLayer.length; j++) {
                outputLayer[i] += hiddenLayer[j] * weightsHiddenOutput[j][i];
            }
        }
        outputLayer = sigmoid(outputLayer);

        return outputLayer;
    }

    public void train(double[][] trainingInputs, double[][] trainingLabels, int epochs) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            for (int i = 0; i < trainingInputs.length; i++) {
                double[] outputs = feedForward(trainingInputs[i]);
                double[] outputErrors = new double[trainingLabels[i].length];
                double[] hiddenErrors = new double[hiddenLayer.length];

                for (int j = 0; j < outputErrors.length; j++) {
                    outputErrors[j] = trainingLabels[i][j] - outputs[j];
                }

                for (int j = 0; j < hiddenLayer.length; j++) {
                    hiddenErrors[j] = 0;
                    for (int k = 0; k < outputErrors.length; k++) {
                        hiddenErrors[j] += outputErrors[k] * weightsHiddenOutput[j][k];
                    }
                }

                // Backpropagation
                for (int j = 0; j < weightsHiddenOutput.length; j++) {
                    for (int k = 0; k < weightsHiddenOutput[j].length; k++) {
                        weightsHiddenOutput[j][k] += learningRate * outputErrors[k] * hiddenLayer[j];
                    }
                }

                for (int j = 0; j < weightsInputHidden.length; j++) {
                    for (int k = 0; k < weightsInputHidden[j].length; k++) {
                        weightsInputHidden[j][k] += learningRate * hiddenErrors[k] * trainingInputs[i][j];
                    }
                }
            }
        }
    }

    public static void main(String[] args) {
        double[][] trainingInputs = { {0, 0}, {0, 1}, {1, 0}, {1, 1} };
        double[][] trainingLabels = { {0}, {1}, {1}, {0} }; // XOR logic
        NeuralNetwork nn = new NeuralNetwork(trainingInputs[0].length, 4, trainingLabels[0].length, 0.1);

        nn.train(trainingInputs, trainingLabels, 10000);

        System.out.println("Prediction for [0, 0]: " + nn.feedForward(new double[]{0, 0})[0]);
        System.out.println("Prediction for [0, 1]: " + nn.feedForward(new double[]{0, 1})[0]);
        System.out.println("Prediction for [1, 0]: " + nn.feedForward(new double[]{1, 0})[0]);
        System.out.println("Prediction for [1, 1]: " + nn.feedForward(new double[]{1, 1})[0]);
    }
}
