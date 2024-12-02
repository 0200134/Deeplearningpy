using System;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace NeuralNetworkExample
{
    class Program
    {
        public class InputData
        {
            [LoadColumn(0, 9)]
            [VectorType(10)]
            public float[] Features { get; set; }

            [LoadColumn(10)]
            public float Label { get; set; }
        }

        public class OutputData
        {
            [ColumnName("Score")]
            public float Score { get; set; }
        }

        static void Main(string[] args)
        {
            var context = new MLContext();

            // Load data from file
            var dataPath = "path_to_your_data.csv";
            var dataView = context.Data.LoadFromTextFile<InputData>(dataPath, hasHeader: true, separatorChar: ',');

            // Define data preparation pipeline
            var pipeline = context.Transforms.Concatenate("Features", new[] { "Features" })
                              .Append(context.Transforms.NormalizeMinMax("Features"))
                              .Append(context.Transforms.Conversion.MapValueToKey("Label"))
                              .Append(context.Model.LoadBinaryTrainer("Label", "Features"))
                              .Append(context.Transforms.Conversion.MapKeyToValue("Score"));

            // Train the model
            var model = pipeline.Fit(dataView);

            // Use the model for predictions
            var predictionEngine = context.Model.CreatePredictionEngine<InputData, OutputData>(model);

            // Example prediction
            var inputData = new InputData { Features = new float[10] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 } };
            var prediction = predictionEngine.Predict(inputData);
            Console.WriteLine($"Prediction: {prediction.Score}");
        }
    }
}
