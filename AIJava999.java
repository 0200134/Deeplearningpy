import ai.djl.Model;
import ai.djl.ModelException;
import ai.djl.basicdataset.nlp.ImdbReview;
import ai.djl.engine.Engine;
import ai.djl.engine.TrainingConfig;
import ai.djl.metric.Metrics;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.recurrent.LSTM;
import ai.djl.nn.core.Embedding;
import ai.djl.nn.core.Linear;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.Trainer;
import ai.djl.training.dataset.Batch;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.listener.EpochTrainingListener;
import ai.djl.training.listener.SaveModelTrainingListener;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Adam;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;

import java.io.IOException;
import java.nio.file.Paths;

public class TextClassificationLSTM {
    public static void main(String[] args) throws IOException, TranslateException, ModelException {
        // Load IMDB dataset
        ImdbReview imdb = ImdbReview.builder()
                .setSampling(32, true)
                .optUsage(Dataset.Usage.TRAIN)
                .build();
        imdb.prepare(new ProgressBar());

        // Define the LSTM model
        Block modelBlock = new SequentialBlock()
                .add(new Embedding.Builder().setEmbeddingSize(128).optPaddingIndex(0).build())
                .add(new LSTM.Builder().setStateSize(128).setNumLayers(2).build())
                .add(Linear.builder().setUnits(128).build())
                .add(Linear.builder().setUnits(2).build());

        Model model = Model.newInstance("text-classification-lstm");
        model.setBlock(modelBlock);

        // Learning rate scheduler
        LearningRateScheduler scheduler = LearningRateScheduler.builder()
                .optWarmupSteps(1000)
                .optBeginLearningRate(0.001f)
                .optEndLearningRate(0.0001f)
                .optDecaySteps(10000)
                .optDecayFactor(0.95f)
                .build();

        // Training configuration
        DefaultTrainingConfig config = new DefaultTrainingConfig(Loss.softmaxCrossEntropyLoss())
                .optOptimizer(Adam.builder().optLearningRateTracker(scheduler).build())
                .addTrainingListeners(new SaveModelTrainingListener("build/model"))
                .addTrainingListeners(new EpochTrainingListener());

        Trainer trainer = model.newTrainer(config);
        trainer.setMetrics(new Metrics());
        Shape inputShape = new Shape(32, 128); // batch size and sequence length
        trainer.initialize(inputShape);

        // Train the model
        for (int epoch = 0; epoch < 20; epoch++) {
            for (Batch batch : trainer.iterateDataset(imdb)) {
                try (NDList data = batch.getData();
                     NDList labels = batch.getLabels()) {
                    trainer.trainBatch(data, labels);
                    trainer.step();
                }
                batch.close();
            }
            System.out.println("Epoch " + epoch + " completed.");
        }

        // Save the model
        model.save(Paths.get("build/model"), "text-classification-lstm");
        System.out.println("Model saved.");
    }
}
