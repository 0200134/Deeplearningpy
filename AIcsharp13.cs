using System;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace NeuralNetworkExample
{
    class Program
    {
        static void Main(string[] args)
        {
            // Create a new ML context
            MLContext mlContext = new MLContext();

            // Load data
            var dataPath = "path_to_your_data.csv";
            IDataView dataView = mlContext.Data.LoadFromTextFile<ModelInput>(dataPath, hasHeader: true, separatorChar: ',');

            // Define data preparation and training pipeline
            var pipeline = mlContext.Transforms.Conversion.MapValueToKey("Label")
                .Append(mlContext.Transforms.Concatenate("Features", "Feature1", "Feature2", "Feature3"))
                .Append(mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy("Label", "Features"))
                .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            // Train the model
            var model = pipeline.Fit(dataView);

            // Use the model for predictions
            var predictionEngine = mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(model);

            var sampleData = new ModelInput { Feature1 = 1.5f, Feature2 = 2.3f, Feature3 = 3.7f };
            var prediction = predictionEngine.Predict(sampleData);

            Console.WriteLine($"Predicted Label: {prediction.PredictedLabel}");
        }
    }

    public class ModelInput
    {
        [LoadColumn(0)]
        public float Feature1 { get; set; }

        [LoadColumn(1)]
        public float Feature2 { get; set; }

        [LoadColumn(2)]
        public float Feature3 { get; set; }

        [LoadColumn(3)]
        public string Label { get; set; }
    }

    public class ModelOutput
    {
        [ColumnName("PredictedLabel")]
        public string PredictedLabel { get; set; }
    }
}
