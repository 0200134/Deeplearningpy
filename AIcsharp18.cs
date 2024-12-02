using System;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace NeuralNetworkExample
{
    public class Program
    {
        public static void Main(string[] args)
        {
            var mlContext = new MLContext();

            // Load the data
            var dataPath = "path/to/your/data.csv";
            var dataView = mlContext.Data.LoadFromTextFile<InputData>(dataPath, separatorChar: ',', hasHeader: true);

            // Define the data preparation and training pipeline
            var pipeline = mlContext.Transforms.Conversion.MapValueToKey("Label")
                .Append(mlContext.Transforms.Concatenate("Features", "Feature1", "Feature2"))
                .Append(mlContext.Transforms.NormalizeMinMax("Features"))
                .Append(mlContext.MulticlassClassification.Trainers.LbfgsMaximumEntropy("Label", "Features"))
                .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            // Train the model
            var model = pipeline.Fit(dataView);

            // Make predictions
            var predictionEngine = mlContext.Model.CreatePredictionEngine<InputData, OutputData>(model);
            var prediction = predictionEngine.Predict(new InputData { Feature1 = 1.0f, Feature2 = 2.0f });

            Console.WriteLine($"Predicted label: {prediction.PredictedLabel}");
        }

        public class InputData
        {
            [LoadColumn(0)]
            public float Feature1 { get; set; }

            [LoadColumn(1)]
            public float Feature2 { get; set; }

            [LoadColumn(2)]
            public string Label { get; set; }
        }

        public class OutputData
        {
            [ColumnName("PredictedLabel")]
            public string PredictedLabel { get; set; }
        }
    }
}
