import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.File;
import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader;
import org.datavec.api.split.FileSplit;

public class LSTMTimeSeriesForecasting {

    public static void main(String[] args) throws Exception {
        int miniBatchSize = 32;
        int numFeatures = 1; // Number of features (e.g., stock prices)
        int numTimesteps = 50; // Number of timesteps in each sequence
        int numEpochs = 100;
        
        // Load and normalize the data
        CSVSequenceRecordReader trainReader = new CSVSequenceRecordReader(0, ",");
        trainReader.initialize(new FileSplit(new File("path_to_your_training_data.csv")));
        DataSetIterator trainData = new RecordReaderDataSetIterator(trainReader, miniBatchSize, -1, numFeatures);
        
        CSVSequenceRecordReader testReader = new CSVSequenceRecordReader(0, ",");
        testReader.initialize(new FileSplit(new File("path_to_your_testing_data.csv")));
        DataSetIterator testData = new RecordReaderDataSetIterator(testReader, miniBatchSize, -1, numFeatures);
        
        NormalizerMinMaxScaler normalizer = new NormalizerMinMaxScaler();
        normalizer.fit(trainData);  // Collects statistics (e.g., min/max) from the training data
        trainData.setPreProcessor(normalizer);
        testData.setPreProcessor(normalizer);

        // Build the LSTM network
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .updater(new Adam(0.001))
                .list()
                .layer(new LSTM.Builder().nIn(numFeatures).nOut(100)
                        .activation(Activation.TANH).build())
                .layer(new LSTM.Builder().nOut(50)
                        .activation(Activation.TANH).build())
                .layer(new RnnOutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .activation(Activation.IDENTITY).nOut(numFeatures).build())
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(10));

        // Train the model
        for (int epoch = 0; epoch < numEpochs; epoch++) {
            model.fit(trainData);
            System.out.println("Epoch " + epoch + " complete.");
        }

        // Evaluate the model
        testData.reset();
        DataSet testSet = testData.next();
        INDArray features = testSet.getFeatures();
        INDArray labels = testSet.getLabels();
        INDArray predicted = model.output(features);
        
        // Print results (add more sophisticated evaluation if needed)
        System.out.println("Features: " + features);
        System.out.println("Labels: " + labels);
        System.out.println("Predicted: " + predicted);
    }
}
