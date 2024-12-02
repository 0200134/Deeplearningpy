public class Perceptron {
    private double[] weights;
    private double learningRate;

    public Perceptron(int inputSize, double learningRate) {
        weights = new double[inputSize + 1]; // Including bias weight
        this.learningRate = learningRate;
    }

    private double activationFunction(double sum) {
        return sum >= 0 ? 1 : 0;
    }

    public int predict(double[] inputs) {
        double sum = weights[weights.length - 1]; // Bias weight
        for (int i = 0; i < inputs.length; i++) {
            sum += inputs[i] * weights[i];
        }
        return (int) activationFunction(sum);
    }

    public void train(double[][] trainingInputs, int[] labels, int epochs) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            for (int i = 0; i < trainingInputs.length; i++) {
                int prediction = predict(trainingInputs[i]);
                int error = labels[i] - prediction;
                for (int j = 0; j < weights.length - 1; j++) {
                    weights[j] += learningRate * error * trainingInputs[i][j];
                }
                weights[weights.length - 1] += learningRate * error; // Adjust bias weight
            }
        }
    }

    public static void main(String[] args) {
        double[][] trainingInputs = { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } };
        int[] labels = { 0, 0, 0, 1 }; // AND gate logic
        Perceptron perceptron = new Perceptron(trainingInputs[0].length, 0.1);

        perceptron.train(trainingInputs, labels, 1000);

        System.out.println("Prediction for [0, 0]: " + perceptron.predict(new double[] { 0, 0 }));
        System.out.println("Prediction for [0, 1]: " + perceptron.predict(new double[] { 0, 1 }));
        System.out.println("Prediction for [1, 0]: " + perceptron.predict(new double[] { 1, 0 }));
        System.out.println("Prediction for [1, 1]: " + perceptron.predict(new double[] { 1, 1 }));
    }
}
