import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.ListDataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

import javax.swing.*;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class HousePricePrediction {

    public static void main(String[] args) {
        // Example data: house sizes, number of bedrooms, age of the house, distance from city center, and prices
        double[][] houseData = {
                {1400, 3, 20, 5}, {1600, 4, 15, 3}, {1700, 3, 10, 8}, {1875, 4, 5, 2},
                {1100, 2, 30, 10}, {1550, 3, 18, 6}, {2350, 4, 8, 1}, {2450, 4, 12, 4},
                {1425, 3, 25, 7}, {1700, 3, 9, 3}
        };
        double[] housePrices = {245000, 312000, 279000, 308000, 199000, 219000, 405000, 324000, 319000, 299000};

        INDArray features = Nd4j.create(houseData);
        INDArray labels = Nd4j.create(housePrices, new long[]{housePrices.length, 1});

        DataSet dataSet = new DataSet(features, labels);

        // Normalize the data
        NormalizerStandardize normalizer = new NormalizerStandardize();
        normalizer.fit(dataSet);
        normalizer.transform(dataSet);

        // Split the data into training and testing sets
        DataSet[] splitData = dataSet.splitTestAndTrain(0.8, new Random(123));
        DataSet trainingData = splitData[0];
        DataSet testData = splitData[1];

        ListDataSetIterator<DataSet> trainIterator = new ListDataSetIterator<>(trainingData.asList(), 5);
        ListDataSetIterator<DataSet> testIterator = new ListDataSetIterator<>(testData.asList(), 5);

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .updater(new Adam())
                .list()
                .layer(new DenseLayer.Builder().nIn(4).nOut(128).activation(Activation.RELU).build())
                .layer(new DenseLayer.Builder().nIn(128).nOut(64).activation(Activation.RELU).build())
                .layer(new DenseLayer.Builder().nIn(64).nOut(32).activation(Activation.RELU).build())
                .layer(new OutputLayer.Builder().nIn(32).nOut(1).activation(Activation.IDENTITY)
                        .lossFunction(LossFunctions.LossFunction.MSE).build())
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(10));

        // Train the model
        model.fit(trainIterator, 200);

        // Evaluate the model
        double testMSE = model.evaluateRegression(testIterator).meanSquaredError(0);
        System.out.println("Test MSE: " + testMSE);

        // Predict with the model
        INDArray sample = Nd4j.create(new double[]{1500, 3, 20, 5}, new long[]{1, 4});
        INDArray predicted = model.output(sample);
        System.out.println("Predicted price for a house of 1500 sq ft, 3 bedrooms, 20 years old, 5 km from city center: " + predicted.getDouble(0));

        // Plotting the results
        plotResults(houseData, housePrices, model, normalizer);
    }

    private static void plotResults(double[][] houseData, double[] housePrices, MultiLayerNetwork model, NormalizerStandardize normalizer) {
        XYSeries series = new XYSeries("Predicted vs Actual Prices");

        for (int i = 0; i < houseData.length; i++) {
            INDArray sample = Nd4j.create(houseData[i]);
            normalizer.transform(sample);
            double actualPrice = housePrices[i];
            double predictedPrice = model.output(sample).getDouble(0);
            series.add(actualPrice, predictedPrice);
        }

        XYSeriesCollection dataset = new XYSeriesCollection(series);
        JFreeChart chart = ChartFactory.createScatterPlot("House Prices", "Actual Price", "Predicted Price", dataset);

        SwingUtilities.invokeLater(() -> {
            JFrame frame = new JFrame("House Price Prediction");
            frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
            frame.add(new ChartPanel(chart));
            frame.pack();
            frame.setVisible(true);
        });
    }
}
