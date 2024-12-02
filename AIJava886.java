import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.optimize.listeners.CheckpointListener;
import org.nd4j.evaluation.regression.RegressionEvaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.ListDataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.schedule.ScheduleType;
import org.nd4j.linalg.schedule.StepSchedule;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

import javax.swing.*;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.Random;

public class AdvancedHousePricePrediction {

    public static void main(String[] args) throws IOException {
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
                .updater(new Adam(new StepSchedule(ScheduleType.ITERATION, 0.01, 0.5, 10000)))
                .list()
                .layer(new DenseLayer.Builder().nIn(4).nOut(256).activation(Activation.RELU).build())
                .layer(new DenseLayer.Builder().nIn(256).nOut(128).activation(Activation.RELU).build())
                .layer(new DenseLayer.Builder().nIn(128).nOut(64).activation(Activation.RELU).build())
                .layer(new OutputLayer.Builder().nIn(64).nOut(1).activation(Activation.IDENTITY)
                        .lossFunction(LossFunctions.LossFunction.MSE).build())
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

        // Advanced logging and monitoring setup
        UIServer uiServer = UIServer.getInstance();
        InMemoryStatsStorage statsStorage = new InMemoryStatsStorage();
        uiServer.attach(statsStorage);
        model.setListeners(new StatsListener(statsStorage), new ScoreIterationListener(10), new CheckpointListener.Builder("checkpoints").saveEveryNIterations(100).build());

        // Train the model with early stopping
        EarlyStoppingConfiguration<MultiLayerNetwork> esConf = new EarlyStoppingConfiguration.Builder<MultiLayerNetwork>()
                .epochTerminationConditions(new MaxEpochsTerminationCondition(200))
                .iterationTerminationConditions(new ScoreImprovementEpochTerminationCondition(5))
                .evaluateEveryNEpochs(1)
                .scoreCalculator(new DataSetLossCalculator(testIterator, true))
                .modelSaver(new LocalFileModelSaver("EarlyStoppingModels"))
                .build();

        EarlyStoppingTrainer trainer = new EarlyStoppingTrainer(esConf, model, trainIterator);
        EarlyStoppingResult<MultiLayerNetwork> result = trainer.fit();

        System.out.println("Termination reason: " + result.getTerminationReason());
        System.out.println("Termination details: " + result.getTerminationDetails());
        System.out.println("Total epochs: " + result.getTotalEpochs());
        System.out.println("Best epoch number: " + result.getBestModelEpoch());
        System.out.println("Score at best epoch: " + result.getBestModelScore());

        // Retrieve the best model
        MultiLayerNetwork bestModel = result.getBestModel();

        // Evaluate the model
        RegressionEvaluation evaluation = bestModel.evaluateRegression(testIterator);
        System.out.println("Test MSE: " + evaluation.meanSquaredError());
        System.out.println("Test MAE: " + evaluation.meanAbsoluteError());
        System.out.println("Test R^2: " + evaluation.rSquared());

        // Save the best model
        File locationToSave = new File("BestHousePricePredictionModel.zip");
        bestModel.save(locationToSave, true);

        // Load the best model
        MultiLayerNetwork restoredModel = MultiLayerNetwork.load(locationToSave, true);

        // Predict with the model
        INDArray sample = Nd4j.create(new double[]{1500, 3, 20, 5}, new long[]{1, 4});
        normalizer.transform(sample);
        INDArray predicted = restoredModel.output(sample);
        System.out.println("Predicted price for a house of 1500 sq ft, 3 bedrooms, 20 years old, 5 km from city center: " + predicted.getDouble(0));

        // Plotting the results
        plotResults(houseData, housePrices, restoredModel, normalizer);
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
